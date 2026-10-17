"""Command-line interface for ragsec."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ragsec.attacks import ATTACK_REGISTRY, AttackTarget, get_attack
from ragsec.defenses import MonitoringLayer
from ragsec.eval import (
    DEFENSE_REGISTRY,
    ExperimentConfig,
    ExperimentRunner,
    build_pipeline,
    format_summary_table,
    load_dataset,
)
from ragsec.eval.runner import decoy_answers, new_signing_key, sign_documents


def _load_config(path: str | None) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path) if path else ExperimentConfig()


def _ingest_clean_corpus(config: ExperimentConfig, dataset: Any):
    """Build the demo pipeline and ingest clean data through its optional trusted signer."""
    signing_key = new_signing_key() if config.sign_corpus else None
    pipeline = build_pipeline(config, signing_key=signing_key)
    documents = (
        sign_documents(dataset.documents, signing_key)
        if signing_key is not None
        else dataset.documents
    )
    pipeline.ingest(documents)
    return pipeline


def cmd_list(args: argparse.Namespace) -> int:
    print("Attacks:")
    for name in sorted(ATTACK_REGISTRY):
        print(f"  {name}")
    print("Defenses:")
    for name in sorted(DEFENSE_REGISTRY):
        print(f"  {name}")
    return 0


def cmd_attack(args: argparse.Namespace) -> int:
    attack = get_attack(args.name, **json.loads(args.params))
    target = AttackTarget(args.query, args.target)
    for index, doc in enumerate(attack.generate_poison([target], args.n), 1):
        print(f"[{index}] {doc.text}")
    print(f"Activation query: {attack.poison_query(target)}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_output:
        config.output_dir = None
    results = ExperimentRunner(config).run()
    print(format_summary_table(results))
    if config.output_dir:
        print(f"\nResults written to {Path(config.output_dir) / config.name}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    dataset = load_dataset(config.dataset)
    pipeline = _ingest_clean_corpus(config, dataset)
    query = args.question
    if args.attack:
        attack = get_attack(args.attack)
        target = AttackTarget(args.question, args.target)
        pipeline.ingest(attack.generate_poison([target], args.n))
        query = attack.poison_query(target)
    response = pipeline.query(query)
    print(f"Query : {query}")
    print(f"Answer: {response.answer}")
    print(f"Generator: {response.metadata['generator']}  "
          f"candidates={response.metadata['num_retrieved']} used={response.metadata['num_used']}")
    print("\nUsed passages:")
    for rc in response.used:
        tag = " [POISON]" if rc.chunk.is_poison else ""
        print(f"  #{rc.rank} score={rc.score:.3f} source={rc.chunk.provenance.source_id}{tag}")
        print(f"     {rc.chunk.text}")
    flagged = [rc for rc in response.filtered if rc.flagged]
    if flagged:
        print("\nFiltered passages:")
        for rc in flagged:
            tag = " [POISON]" if rc.chunk.is_poison else ""
            print(f"  #{rc.rank} score={rc.score:.3f}{tag} reasons={rc.flag_reasons}")
            print(f"     {rc.chunk.text}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    dataset = load_dataset(config.dataset)
    pipeline = _ingest_clean_corpus(config, dataset)
    monitor = next((d for d in pipeline.defenses if isinstance(d, MonitoringLayer)), None)
    if monitor is None:
        monitor = MonitoringLayer(window_size=len(dataset.qa))
        pipeline.add_defense(monitor)
    questions = [ex.question for ex in dataset.qa]
    monitor.register_benchmarks(questions)
    for ex in dataset.qa[:5]:
        monitor.register_canary(ex.question, ex.answer)

    for question in questions:
        pipeline.query(question)
    monitor.flush()
    print(f"Baseline window recorded ({len(questions)} queries), alerts: {len(monitor.alerts)}")

    attack = get_attack(args.attack)
    decoys = decoy_answers(dataset.qa, random.Random(config.seed))
    targets = [
        AttackTarget(ex.question, ex.answer if attack.suppresses_answer else decoys[ex.question])
        for ex in dataset.qa
    ]
    result = pipeline.ingest(attack.generate_poison(targets, args.n))
    print(f"Injected {result.accepted_documents} poison documents via {attack.name}")
    for target in targets:
        pipeline.query(attack.poison_query(target))
    monitor.flush()
    monitor.check_benchmarks()
    monitor.check_canaries()

    counts = Counter(alert.kind for alert in monitor.alerts)
    print("\nAlerts:")
    if not counts:
        print("  none")
    for kind, count in counts.items():
        first = next(a for a in monitor.alerts if a.kind == kind)
        print(f"  {kind:<16} x{count:<3} [{first.severity}] {first.message}")
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    from ragsec.eval.suite import STUDIES, run_suite

    rows = run_suite(
        studies=args.studies or None,
        datasets=args.datasets or None,
        seeds=args.seeds or None,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        models=args.models or None,
        base_url=args.base_url or None,
        api_key_env=args.api_key_env or None,
        queries=args.queries or None,
    )
    for study in args.studies or list(STUDIES):
        print(f"{study}: {len(rows.get(study, []))} rows -> {args.output_dir}/{study}.jsonl")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from ragsec.eval.report import build_report
    from ragsec.eval.texcheck import check_manuscript

    output_dir = Path(args.output_dir)
    written = build_report(Path(args.results_dir), output_dir)
    for path in written:
        print(path)
    if args.no_check:
        return 0
    root = Path(args.manuscript).parent
    sources = [
        Path(args.manuscript),
        output_dir / "body.tex",
        *sorted((output_dir / "tables").glob("*.tex")),
    ]
    report = check_manuscript(sources, root, output_dir, output_dir / "bibliography.tex")
    for problem in report.problems:
        print(f"  manuscript: {problem}")
    if args.strict:
        for warning in report.warnings:
            print(f"  manuscript (minor): {warning}")
    # A partially-run suite legitimately leaves tables and figures missing, so this reports
    # rather than fails unless the caller asked for strictness.
    return 1 if (args.strict and report.problems) else 0


def cmd_manuscript(args: argparse.Namespace) -> int:
    from ragsec.eval.manuscript import build

    summary = build(Path(args.project_root), Path(args.output), journal=args.journal)
    print(f"wrote {args.output}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if summary["em_dashes"]:
        print("  WARNING: em dashes present; this journal style forbids them")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragsec", description="RAG poisoning attacks and defenses research toolkit"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available attacks and defenses")
    p_list.set_defaults(func=cmd_list)

    p_attack = sub.add_parser("attack", help="craft poisoned passages for a target query")
    p_attack.add_argument("--name", required=True, choices=sorted(ATTACK_REGISTRY))
    p_attack.add_argument("--query", required=True)
    p_attack.add_argument("--target", required=True, help="adversarial (or suppressed) answer")
    p_attack.add_argument("-n", type=int, default=1, help="passages per target")
    p_attack.add_argument("--params", default="{}", help="JSON kwargs for the attack")
    p_attack.set_defaults(func=cmd_attack)

    p_eval = sub.add_parser("evaluate", help="run an experiment config")
    p_eval.add_argument("config", nargs="?", help="YAML config (default: undefended toy run)")
    p_eval.add_argument("--output-dir")
    p_eval.add_argument("--no-output", action="store_true", help="do not write result files")
    p_eval.set_defaults(func=cmd_evaluate)

    p_query = sub.add_parser("query", help="answer one question, optionally under attack")
    p_query.add_argument("question")
    p_query.add_argument("--config")
    p_query.add_argument("--attack", choices=sorted(ATTACK_REGISTRY))
    p_query.add_argument("--target", default="DECOY")
    p_query.add_argument("-n", type=int, default=1)
    p_query.set_defaults(func=cmd_query)

    p_mon = sub.add_parser("monitor", help="demonstrate anomaly monitoring across a poisoning")
    p_mon.add_argument("--config")
    p_mon.add_argument("--attack", default="context_flooding", choices=sorted(ATTACK_REGISTRY))
    p_mon.add_argument("-n", type=int, default=3)
    p_mon.set_defaults(func=cmd_monitor)

    p_suite = sub.add_parser("suite", help="run the full experiment suite reported in the paper")
    p_suite.add_argument("--studies", nargs="*", default=[], help="subset of studies to run")
    p_suite.add_argument("--datasets", nargs="*", default=[], help="subset of datasets")
    p_suite.add_argument("--seeds", nargs="*", type=int, default=[], help="random seeds")
    p_suite.add_argument("--output-dir", default="results/paper-pinned")
    p_suite.add_argument("--no-resume", action="store_true", help="recompute finished cells")
    p_suite.add_argument(
        "--models", nargs="*", default=[],
        help="models for the confirmation study (default: gpt-4o-mini gpt-4o)",
    )
    p_suite.add_argument(
        "--base-url", default="",
        help=(
            "OpenAI-compatible endpoint for the confirmation study, e.g. "
            "https://api.groq.com/openai/v1 or http://localhost:11434/v1 for Ollama"
        ),
    )
    p_suite.add_argument(
        "--api-key-env", default="",
        help="environment variable holding the API key (default: OPENAI_API_KEY)",
    )
    p_suite.add_argument(
        "--queries", type=int, default=0,
        help=(
            "queries per corpus in the confirmation study (default 100). Lower it to stay "
            "inside a provider's free-tier request cap"
        ),
    )
    p_suite.set_defaults(func=cmd_suite)

    p_fig = sub.add_parser("figures", help="build the paper's tables and figures from results")
    p_fig.add_argument("--results-dir", default="results/paper-pinned")
    p_fig.add_argument("--output-dir", default="paper")
    p_fig.add_argument("--manuscript", default="paper/preview.tex")
    p_fig.add_argument("--no-check", action="store_true", help="skip the manuscript checks")
    p_fig.add_argument(
        "--strict", action="store_true", help="also report minor findings and exit non-zero"
    )
    p_fig.set_defaults(func=cmd_figures)

    p_ms = sub.add_parser(
        "manuscript", help="assemble the single self-contained MDPI submission .tex"
    )
    p_ms.add_argument("--project-root", default=".")
    p_ms.add_argument("--output", default="paper/generated/template.tex")
    p_ms.add_argument("--journal", default="information")
    p_ms.set_defaults(func=cmd_manuscript)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
