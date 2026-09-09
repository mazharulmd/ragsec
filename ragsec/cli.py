"""Command-line interface for ragsec."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

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
from ragsec.eval.runner import decoy_answers


def _load_config(path: str | None) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path) if path else ExperimentConfig()


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
    pipeline = build_pipeline(config)
    pipeline.ingest(dataset.documents)
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
    pipeline = build_pipeline(config)
    monitor = next((d for d in pipeline.defenses if isinstance(d, MonitoringLayer)), None)
    if monitor is None:
        monitor = MonitoringLayer(window_size=len(dataset.qa))
        pipeline.add_defense(monitor)
    pipeline.ingest(dataset.documents)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
