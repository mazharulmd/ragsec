"""Turn suite results into the paper's tables and figures.

Everything the manuscript reports is produced here from ``results/paper/*.jsonl``, so no number
in the paper is transcribed by hand. Tables are emitted as LaTeX fragments that the manuscript
``\\input``s; figures are written as vector PDF (for the submission) alongside PNG (for review).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from ragsec.eval.suite import ATTACK_LABELS, CONFIG_LABELS, EVASION_LABELS

Row = dict[str, Any]

# ---------------------------------------------------------------------------
# House style
#
# One accent colour per semantic role, colour-blind safe (Okabe-Ito derived), and every figure
# also encodes its distinctions in position, marker or hatch so it survives greyscale printing.
# ---------------------------------------------------------------------------

C_ATTACK = "#c0392b"      # adversary / risk
C_DEFENSE = "#1a6fa3"     # defended configuration
C_SIGNED = "#117a54"      # cryptographically bound configuration
C_NEUTRAL = "#6b6b6b"
C_UTILITY = "#b8770a"     # benign utility
C_GRID = "#d8d8d8"

CONFIG_COLORS = {
    "none": C_ATTACK,
    "L1": "#8e6fb5",
    "L2": "#4f9ed6",
    "L3": "#e0913a",
    "L4": C_NEUTRAL,
    "full": C_DEFENSE,
    "full_signed": C_SIGNED,
}

ATTACK_ORDER = [
    "poisoned_rag",
    "trigger_backdoor",
    "prompt_injection",
    "context_flooding",
    "adaptive_poison",
]
CONFIG_ORDER = ["none", "L1", "L2", "L3", "L4", "full", "full_signed"]
DATASET_LABELS = {"squad": "SQuAD", "hotpotqa": "HotpotQA", "toy": "Toy corpus"}


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": C_GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
        }
    )


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------


def load_rows(results_dir: Path, study: str) -> list[Row]:
    path = results_dir / f"{study}.jsonl"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite(values: Iterable[Any]) -> list[float]:
    out = []
    for v in values:
        if v is None:
            continue
        f = float(v)
        if math.isfinite(f):
            out.append(f)
    return out


def agg(
    rows: Sequence[Row],
    keys: Sequence[str],
    field: str,
    reduce: Callable[[Sequence[float]], float] = statistics.fmean,
) -> dict[tuple, float]:
    """Reduce ``field`` over the seeds sharing each combination of ``keys``."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        f = float(value)
        if math.isfinite(f):
            buckets[tuple(row[k] for k in keys)].append(f)
    return {k: reduce(v) for k, v in buckets.items() if v}


def agg_spread(rows: Sequence[Row], keys: Sequence[str], field: str) -> dict[tuple, float]:
    """Half-width of the across-seed range, used as the seed-variability error bar."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        f = float(value)
        if math.isfinite(f):
            buckets[tuple(row[k] for k in keys)].append(f)
    return {k: (max(v) - min(v)) / 2.0 for k, v in buckets.items() if v}


def seed_summary(
    rows: Sequence[Row],
    keys: Sequence[str],
    rate_field: str = "attack_success_rate",
    n_field: str = "n_attacked",
) -> dict[tuple, tuple[float, float, float]]:
    """Return pooled rate and minimum/maximum per-seed rates for each key.

    Rows retain only seed-level aggregates, not per-question outcomes. We therefore summarize
    between-seed variability rather than treating repeated questions as independent Bernoulli
    trials. The point estimate pools numerator and denominator across seeds; when multiple
    attack rows share a seed and key, ``n_field`` weights their rates within that seed. The
    endpoints are the observed seed range, not a confidence interval. Pass
    ``rate_field="conditional_asr"`` with ``n_field="n_baseline_correct"`` for
    attack-attributable rates.
    """
    by_seed: dict[tuple[tuple, Any], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        rate = row.get(rate_field)
        count = row.get(n_field) or 0
        if rate is None or not count:
            continue
        key = tuple(row[k] for k in keys)
        seed_key = (key, row.get("seed"))
        by_seed[seed_key][0] += float(rate) * int(count)
        by_seed[seed_key][1] += int(count)
    per_key: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for (key, _seed), (weighted_sum, total) in by_seed.items():
        if total:
            per_key[key].append((weighted_sum, total))
    out = {}
    for key, seed_counts in per_key.items():
        rates = [successes / total for successes, total in seed_counts]
        out[key] = (
            sum(successes for successes, _total in seed_counts)
            / sum(total for _successes, total in seed_counts),
            min(rates),
            max(rates),
        )
    return out


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------



#: TeX control words are sequences of letters only, so a macro name may not contain a digit:
#: ``\casrStaticSquadL3`` parses as ``\casrStaticSquadL`` followed by a literal "3" and fails
#: with an undefined-control-sequence error. Numbers appearing in generated names (layer
#: indices, budgets) are therefore spelled out.
_NUMBER_WORDS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 20: "Twenty",
}
_DIGIT_WORDS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}


def macro_name(name: str) -> str:
    """Return ``name`` as a legal TeX control word: letters only."""

    def spell(match: re.Match[str]) -> str:
        digits = match.group()
        whole = _NUMBER_WORDS.get(int(digits))
        return whole if whole else "".join(_DIGIT_WORDS[d] for d in digits)

    return re.sub(r"\d+", spell, name).replace("_", "")


def fmt(value: float | None, places: int = 2, dash: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return dash
    return f"{value:.{places}f}"


def latex_table(
    caption: str,
    label: str,
    header: Sequence[str],
    body: Sequence[Sequence[str]],
    column_spec: str | None = None,
    note: str = "",
    wide: bool | None = None,
) -> str:
    """Render a booktabs table.

    ``wide`` wraps the table in MDPI's ``adjustwidth`` so that a many-column table may use the
    margins instead of overflowing the text block; it defaults to on above seven columns, which
    is roughly where a numeric table stops fitting the journal's measure.
    """
    spec = column_spec or ("l" + "X" * (len(header) - 1))
    env = "tabularx" if "X" in spec else "tabular"
    width = "{\\textwidth}" if env == "tabularx" else ""
    if wide is None:
        wide = len(header) > 7
    lines = ["\\begin{table}[H]"]
    if wide:
        lines += ["\\begin{adjustwidth}{-\\extralength}{0cm}", "\\footnotesize"]
    else:
        lines.append("\\small")
    lines += [
        f"\\caption{{{caption}\\label{{{label}}}}}",
        f"\\begin{{{env}}}{width}{{{spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in header) + "\\\\",
        "\\midrule",
    ]
    lines += [" & ".join(cells) + "\\\\" for cells in body]
    lines += ["\\bottomrule", f"\\end{{{env}}}"]
    if note:
        lines.append(f"\\noindent\\footnotesize {note}")
    if wide:
        lines.append("\\end{adjustwidth}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / f"{name}.png")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_architecture(out_dir: Path) -> Path:
    """Schematic of the pipeline, the four layers, and where the adversary enters."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")
    ax.grid(False)

    LAYER_FACE, PLAIN_FACE = "#e6f0f8", "#f3f3f3"

    def box(x, y, w, h, title, face, edge, caption="", weight="normal", tc="black",
            fontsize=7.0):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.45,rounding_size=1.0",
                facecolor=face, edgecolor=edge, linewidth=1.0,
            )
        )
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fontsize, weight=weight, color=tc, linespacing=1.3)
        if caption:
            ax.text(x + w / 2, y - 1.3, caption, ha="center", va="top",
                    fontsize=5.8, color=C_NEUTRAL, linespacing=1.25, style="italic")

    def arrow(x1, y1, x2, y2, color=C_NEUTRAL, lw=1.1, ls="-"):
        ax.add_patch(
            FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                            color=color, linewidth=lw, linestyle=ls, shrinkA=1, shrinkB=1)
        )

    # --- adversary band -----------------------------------------------------
    box(1, 49, 33, 8,
        "Adversary\nwrites to the knowledge base; cannot modify\n"
        "model weights or the ingestion signing key",
        "#fbeceb", C_ATTACK, fontsize=6.6, tc=C_ATTACK)
    ax.text(38.5, 56.6, "Attack surface", fontsize=7.3, weight="bold", color=C_ATTACK,
            va="top")
    for i, text in enumerate(["A1  targeted poisoning", "A2  trigger backdoor",
                              "A3  indirect prompt injection"]):
        ax.text(38.5, 53.6 - 2.3 * i, text, fontsize=6.4, color=C_ATTACK, va="top")
    for i, text in enumerate(["A4  context flooding",
                              "A5  adaptive (defense-aware) poisoning"]):
        ax.text(69, 53.6 - 2.3 * i, text, fontsize=6.4, color=C_ATTACK, va="top")

    # --- ingestion lane -----------------------------------------------------
    ax.text(0, 44.5, "Ingestion", fontsize=6.6, color=C_NEUTRAL, weight="bold",
            va="bottom")
    box(1, 34, 16, 8, "Document\nsources", PLAIN_FACE, C_NEUTRAL,
        "provenance:\nsource, trust, signature")
    box(22, 34, 17, 8, "L1  Source\nverification", LAYER_FACE, C_DEFENSE,
        "allowlist $\\cdot$ trust $\\cdot$ HMAC\ndedupe $\\cdot$ quota", weight="bold")
    box(44, 34, 15, 8, "Chunk and\nembed", PLAIN_FACE, C_NEUTRAL)
    box(64, 34, 15, 8, "Vector\nindex", PLAIN_FACE, C_NEUTRAL)
    for x1, x2 in [(17, 22), (39, 44), (59, 64)]:
        arrow(x1, 38, x2, 38)
    arrow(19, 49, 9, 42, C_ATTACK, lw=1.3)

    # --- query lane ---------------------------------------------------------
    ax.text(0, 22.5, "Query time", fontsize=6.6, color=C_NEUTRAL, weight="bold",
            va="bottom")
    box(1, 12, 14, 8, "Query", PLAIN_FACE, C_NEUTRAL)
    box(20, 12, 17, 8, "L2  Retrieval\nfiltering", LAYER_FACE, C_DEFENSE,
        "instruction $\\cdot$ echo\noutlier $\\cdot$ naturalness", weight="bold")
    box(42, 12, 17, 8, "L3  Robust\naggregation", LAYER_FACE, C_DEFENSE,
        "isolate $\\cdot$ weighted vote\nabstain on low margin", weight="bold")
    box(64, 12, 14, 8, "Generator\n(LLM)", PLAIN_FACE, C_NEUTRAL)
    box(83, 12, 16, 8, "Answer or\nabstention", PLAIN_FACE, C_NEUTRAL)
    for x1, x2 in [(15, 20), (37, 42), (59, 64), (78, 83)]:
        arrow(x1, 16, x2, 16)

    # over-retrieved candidate set flows from the index into the filter
    arrow(71.5, 34, 71.5, 25.5, C_DEFENSE)
    arrow(71.5, 25.5, 28.5, 25.5, C_DEFENSE)
    arrow(28.5, 25.5, 28.5, 20, C_DEFENSE)
    ax.text(71, 26.5, "top-$mk$ candidate set (over-retrieval enables backfill)",
            ha="right", fontsize=6.3, color=C_DEFENSE, style="italic")

    # --- monitoring ---------------------------------------------------------
    box(60, 0.5, 39, 5.5, "L4  Anomaly monitoring", "#e8f5ef", C_SIGNED,
        weight="bold", fontsize=6.8)
    ax.text(79.5, 3.25 - 2.0, "", ha="center")
    ax.text(79.5, -1.4, "retrieval drift $\\cdot$ ranking shift $\\cdot$ canary queries",
            ha="center", va="top", fontsize=5.8, color=C_NEUTRAL, style="italic")
    arrow(91, 12, 91, 6, C_SIGNED)
    arrow(60, 3.25, 30, 3.25, C_SIGNED, ls=(0, (4, 2)))
    ax.text(45, 4.5, "alerts to operators (detective, not preventive)",
            ha="center", fontsize=6.2, color=C_SIGNED, style="italic")

    return _save(fig, out_dir, "fig1_architecture")


def fig_asr_heatmap(rows: list[Row], out_dir: Path) -> Path:
    """ASR for every (defense configuration, attack) pair, one panel per dataset."""
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    values = seed_summary(rows, ["dataset", "config", "attack"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.9), squeeze=False)
    for ax, dataset in zip(axes[0], datasets, strict=False):
        grid = np.full((len(CONFIG_ORDER), len(ATTACK_ORDER)), np.nan)
        for i, config in enumerate(CONFIG_ORDER):
            for j, attack in enumerate(ATTACK_ORDER):
                hit = values.get((dataset, config, attack))
                if hit is not None:
                    grid[i, j] = hit[0]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(ATTACK_ORDER)))
        ax.set_xticklabels([ATTACK_LABELS[a].replace(" ", "\n") for a in ATTACK_ORDER],
                           fontsize=6.2)
        ax.set_yticks(range(len(CONFIG_ORDER)))
        ax.set_yticklabels(
            [CONFIG_LABELS[c] for c in CONFIG_ORDER] if dataset == datasets[0] else [],
            fontsize=6.6,
        )
        ax.set_title(DATASET_LABELS[dataset], fontsize=8.5)
        ax.grid(False)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isnan(grid[i, j]):
                    continue
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=5.9,
                        color="white" if 0.32 < grid[i, j] < 0.72 else "black")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.022, pad=0.015,
                 label="Attack success rate")
    return _save(fig, out_dir, "fig2_asr_heatmap")


def fig_asr_bars(rows: list[Row], out_dir: Path, dataset: str = "squad") -> Path:
    """Headline comparison with observed min--max seed ranges."""
    import matplotlib.pyplot as plt
    import numpy as np

    subset = [r for r in rows if r["dataset"] == dataset]
    values = seed_summary(subset, ["config", "attack"])
    configs = ["none", "L2", "L3", "full", "full_signed"]
    width = 0.16
    x = np.arange(len(ATTACK_ORDER))
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    for i, config in enumerate(configs):
        heights, lows, highs = [], [], []
        for attack in ATTACK_ORDER:
            hit = values.get((config, attack))
            heights.append(hit[0] if hit else 0.0)
            lows.append(max(0.0, hit[0] - hit[1]) if hit else 0.0)
            highs.append(max(0.0, hit[2] - hit[0]) if hit else 0.0)
        offset = (i - (len(configs) - 1) / 2) * width
        ax.bar(x + offset, heights, width, label=CONFIG_LABELS[config],
               color=CONFIG_COLORS[config], edgecolor="white", linewidth=0.4)
        ax.errorbar(x + offset, heights, yerr=[lows, highs], fmt="none",
                    ecolor="#333333", elinewidth=0.7, capsize=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS[a] for a in ATTACK_ORDER], fontsize=7.4)
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    return _save(fig, out_dir, "fig3_asr_bars")


def fig_evasion(rows: list[Row], out_dir: Path) -> Path:
    """Leave-one-out over evasion techniques: which one does each layer actually stop?"""
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    values = seed_summary(rows, ["dataset", "config", "disabled"])
    order = ["none", *EVASION_LABELS]
    labels = ["All techniques enabled"] + [
        f"without: {EVASION_LABELS[k].lower()}" for k in EVASION_LABELS
    ]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.7), squeeze=False, sharex=True)
    for ax, dataset in zip(axes[0], datasets, strict=False):
        for offset, config, color in ((-0.19, "full", C_DEFENSE), (0.19, "full_signed", C_SIGNED)):
            heights, err_lo, err_hi = [], [], []
            for key in order:
                hit = values.get((dataset, config, key))
                heights.append(hit[0] if hit else 0.0)
                err_lo.append(max(0.0, hit[0] - hit[1]) if hit else 0.0)
                err_hi.append(max(0.0, hit[2] - hit[0]) if hit else 0.0)
            ax.barh(y + offset, heights, 0.36, color=color, label=CONFIG_LABELS[config],
                    edgecolor="white", linewidth=0.4)
            ax.errorbar(heights, y + offset, xerr=[err_lo, err_hi], fmt="none",
                        ecolor="#333333", elinewidth=0.6, capsize=1.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if dataset == datasets[0] else [], fontsize=6.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Attack success rate")
        ax.set_title(DATASET_LABELS[dataset], fontsize=8.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.13),
               fontsize=6.8)
    return _save(fig, out_dir, "fig4_evasion")


def fig_budget(rows: list[Row], out_dir: Path) -> Path:
    """Attack success against the adversary's upload budget."""
    import matplotlib.pyplot as plt

    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    values = seed_summary(rows, ["dataset", "config", "attack", "n_per_target"],
                       "conditional_asr", "n_baseline_correct")
    budgets = sorted({r["n_per_target"] for r in rows})
    series = [
        ("poisoned_rag", "none", C_ATTACK, "o", "-"),
        ("poisoned_rag", "full", C_DEFENSE, "o", "-"),
        ("adaptive_poison", "none", C_ATTACK, "s", "--"),
        ("adaptive_poison", "full", C_DEFENSE, "s", "--"),
        ("adaptive_poison", "full_signed", C_SIGNED, "^", ":"),
    ]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.5), squeeze=False, sharey=True)
    for ax, dataset in zip(axes[0], datasets, strict=False):
        for attack, config, color, marker, ls in series:
            xs, ys = [], []
            for n in budgets:
                hit = values.get((dataset, config, attack, n))
                if hit is not None:
                    xs.append(n)
                    ys.append(hit[0])
            if xs:
                ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                        label=f"{ATTACK_LABELS[attack]} / {CONFIG_LABELS[config]}")
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel("Poisoned documents per target")
        ax.set_ylim(-0.04, 1.06)
        ax.set_title(DATASET_LABELS[dataset], fontsize=8.5)
    axes[0][0].set_ylabel("Attributable attack success rate")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22),
               fontsize=6.8)
    return _save(fig, out_dir, "fig5_budget")


def fig_detection(rows: list[Row], out_dir: Path) -> Path:
    """Empirical ROC of the retrieval filter, traced by sweeping its suspicion threshold."""
    import matplotlib.pyplot as plt

    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    tpr = agg(rows, ["dataset", "attack", "threshold"], "filter_tpr")
    fpr = agg(rows, ["dataset", "attack", "threshold"], "filter_fpr")
    thresholds = sorted({r["threshold"] for r in rows})
    markers = dict(zip(ATTACK_ORDER, ["o", "s", "^", "D", "v"], strict=False))
    colors = dict(zip(ATTACK_ORDER,
                      ["#c0392b", "#8e6fb5", "#4f9ed6", "#e0913a", "#111111"], strict=False))
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.5), squeeze=False, sharey=True)
    for ax, dataset in zip(axes[0], datasets, strict=False):
        ax.plot([0, 1], [0, 1], color=C_GRID, linestyle=(0, (2, 2)), linewidth=0.9, zorder=0)
        for attack in ATTACK_ORDER:
            xs, ys = [], []
            for t in thresholds:
                x = fpr.get((dataset, attack, t))
                y = tpr.get((dataset, attack, t))
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            if xs:
                pairs = sorted(zip(xs, ys, strict=False))
                ax.plot([0, *[p[0] for p in pairs], 1], [0, *[p[1] for p in pairs], 1],
                        marker=markers[attack], color=colors[attack],
                        label=ATTACK_LABELS[attack], markersize=3.6, linewidth=1.3)
        ax.set_xlabel("False-positive rate")
        ax.set_xlim(-0.02, 0.5)
        ax.set_ylim(-0.03, 1.04)
        ax.set_title(DATASET_LABELS[dataset], fontsize=8.5)
    axes[0][0].set_ylabel("True-positive rate")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.16),
               fontsize=6.8)
    return _save(fig, out_dir, "fig6_detection_roc")


def fig_threshold(rows: list[Row], out_dir: Path, dataset: str = "squad") -> Path:
    """What moving the filter's operating point costs in false positives and utility.

    Detection is split by adversary rather than pooled: the whole point is that the threshold
    which catches the static constructions is nowhere near the one that catches the adaptive
    rewrite, and pooling the two would hide exactly that.
    """
    import matplotlib.pyplot as plt

    subset = [r for r in rows if r["dataset"] == dataset]
    thresholds = sorted({r["threshold"] for r in subset})
    static = [a for a in ATTACK_ORDER if a != "adaptive_poison"]
    static_rows = [r for r in subset if r["attack"] in static]
    adaptive_rows = [r for r in subset if r["attack"] == "adaptive_poison"]
    clean_rows = [r for r in subset if r["attack"] == "clean"]

    series = [
        (agg(static_rows, ["threshold"], "filter_tpr"),
         "Filter TPR, static", C_SIGNED, "o", "-"),
        (agg(adaptive_rows, ["threshold"], "filter_tpr"),
         "Filter TPR, adaptive", C_SIGNED, "v", (0, (4, 2))),
        (agg(clean_rows, ["threshold"], "filter_fpr"),
         "Filter FPR, clean traffic", C_ATTACK, "s", "-"),
        (agg(clean_rows, ["threshold"], "clean_accuracy"),
         "Clean accuracy", C_UTILITY, "^", "-"),
        (seed_summary(static_rows, ["threshold"], "conditional_asr", "n_baseline_correct"),
         "Attributable ASR, static", C_DEFENSE, "D", "-"),
        (seed_summary(adaptive_rows, ["threshold"], "conditional_asr", "n_baseline_correct"),
         "Attributable ASR, adaptive", "#111111", "X", (0, (4, 2))),
    ]

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for mapping, label, color, marker, style in series:
        xs = [t for t in thresholds if (t,) in mapping]
        ys = [mapping[(t,)] for t in xs]
        ys = [y[0] if isinstance(y, tuple) else y for y in ys]
        ax.plot(xs, ys, marker=marker, color=color, label=label, linestyle=style)
    ax.set_xlabel("Suspicion threshold $\\tau$")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(ncol=3, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.19))
    return _save(fig, out_dir, "fig7_threshold")


def fig_backfill(
    rows: list[Row],
    out_dir: Path,
    strict_rows: list[Row] | None = None,
    dataset: str = "squad",
) -> Path:
    """Security, utility and latency as over-retrieval increases, at two operating points.

    Backfill can only matter when the filter is actually discarding candidates. At the default
    threshold it discards well under one percent of them, so the left panel is flat and the
    multiplier buys nothing but latency; the right panel repeats the sweep at a threshold
    aggressive enough to remove a large share of the candidate set, which is the regime the
    "filtering needs over-retrieval" argument is really about.
    """
    import matplotlib.pyplot as plt

    panels = [("Default threshold $\\tau=1.0$", rows)]
    if strict_rows:
        panels.append(("Strict threshold $\\tau=0.6$", strict_rows))

    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.7 * len(panels), 2.8), squeeze=False,
        gridspec_kw={"wspace": 0.55},
    )
    static = [a for a in ATTACK_ORDER if a != "adaptive_poison"]
    for ax, (title, source) in zip(axes[0], panels, strict=False):
        subset = [r for r in source if r["dataset"] == dataset]
        multipliers = sorted({r["candidate_multiplier"] for r in subset})
        clean_rows = [r for r in subset if r["attack"] == "clean"]
        asr = seed_summary([r for r in subset if r["attack"] in static],
                        ["candidate_multiplier"], "conditional_asr", "n_baseline_correct")
        acc = agg(clean_rows, ["candidate_multiplier"], "clean_accuracy")
        abstain = agg(clean_rows, ["candidate_multiplier"], "abstention_rate")
        latency = agg(subset, ["candidate_multiplier"], "p95_latency_ms")

        ax.plot(multipliers, [acc.get((m,), float("nan")) for m in multipliers],
                marker="^", color=C_UTILITY, label="Clean accuracy")
        ax.plot(multipliers, [abstain.get((m,), float("nan")) for m in multipliers],
                marker="s", color=C_NEUTRAL, label="Abstention rate")
        ax.plot(multipliers,
                [asr[(m,)][0] if (m,) in asr else float("nan") for m in multipliers],
                marker="o", color=C_ATTACK, label="Attributable ASR, static")
        ax.set_xscale("log")
        ax.set_xticks(multipliers)
        ax.set_xticklabels([str(m) for m in multipliers])
        ax.set_xlabel("Candidate multiplier $m$")
        ax.set_ylim(-0.04, 1.06)
        ax.set_title(title, fontsize=8.5)

        twin = ax.twinx()
        twin.plot(multipliers, [latency.get((m,), float("nan")) for m in multipliers],
                  marker="D", color=C_DEFENSE, linestyle="--", label="p95 latency")
        twin.set_ylabel("p95 latency (ms)", color=C_DEFENSE)
        twin.tick_params(axis="y", colors=C_DEFENSE)
        twin.grid(False)
        twin.spines["right"].set_visible(True)
        twin.spines["right"].set_color(C_DEFENSE)

    axes[0][0].set_ylabel("Rate")
    handles = axes[0][0].get_legend_handles_labels()[0]
    labels = axes[0][0].get_legend_handles_labels()[1]
    import matplotlib.lines as mlines

    handles.append(mlines.Line2D([], [], color=C_DEFENSE, linestyle="--", marker="D"))
    labels.append("p95 latency (right axis)")
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.13),
               fontsize=6.8)
    return _save(fig, out_dir, "fig8_backfill")


def fig_provenance(rows: list[Row], out_dir: Path) -> Path:
    """Ingestion controls of increasing strength against a provenance-spoofing adversary."""
    import matplotlib.pyplot as plt
    import numpy as np

    order = ["open", "trust_only", "allowlist", "quota", "signed"]
    labels = ["None", "Trust", "Allowlist", "Quota", "HMAC"]
    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    values = seed_summary(rows, ["dataset", "variant", "attack"])
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.5), squeeze=False, sharey=True)
    for ax, dataset in zip(axes[0], datasets, strict=False):
        for offset, attack, color, hatch in (
            (-0.2, "poisoned_rag", C_DEFENSE, ""),
            (0.2, "adaptive_poison", C_ATTACK, "///"),
        ):
            heights = [
                values.get((dataset, v, attack), (0.0, 0.0, 0.0))[0] for v in order
            ]
            errs = [
                (
                    max(0.0, values.get((dataset, v, attack), (0.0, 0.0, 0.0))[0]
                        - values.get((dataset, v, attack), (0.0, 0.0, 0.0))[1]),
                    max(0.0, values.get((dataset, v, attack), (0.0, 0.0, 0.0))[2]
                        - values.get((dataset, v, attack), (0.0, 0.0, 0.0))[0]),
                )
                for v in order
            ]
            ax.bar(x + offset, heights, 0.38, color=color, hatch=hatch,
                   edgecolor="white", linewidth=0.5, label=ATTACK_LABELS[attack])
            ax.errorbar(x + offset, heights,
                        yerr=[[e[0] for e in errs], [e[1] for e in errs]],
                        fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.0, rotation=20, ha="right",
                           rotation_mode="anchor")
        ax.set_ylim(0, 1.08)
        ax.set_title(DATASET_LABELS[dataset], fontsize=8.5)
    axes[0][0].set_ylabel("Attack success rate")
    axes[0][0].legend(fontsize=6.8, loc="upper right")
    return _save(fig, out_dir, "fig9_provenance")


def fig_latency(rows: list[Row], out_dir: Path, dataset: str = "squad") -> Path:
    """Where the defended pipeline spends its time."""
    import matplotlib.pyplot as plt
    import numpy as np

    subset = [r for r in rows if r["dataset"] == dataset and r["attack"] != "clean"]
    configs = [c for c in CONFIG_ORDER if any(r["config"] == c for r in subset)]
    stages = [("mean_retrieve_ms", "Retrieval", "#9dc6e0"),
              ("mean_filter_ms", "L2 filtering", C_DEFENSE),
              ("mean_generate_ms", "Generation (incl. L3)", C_UTILITY)]
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    bottom = np.zeros(len(configs))
    for field, label, color in stages:
        values = agg(subset, ["config"], field)
        heights = np.array([values.get((c,), 0.0) for c in configs])
        ax.bar(range(len(configs)), heights, 0.62, bottom=bottom, color=color,
               label=label, edgecolor="white", linewidth=0.5)
        bottom += heights
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([CONFIG_LABELS[c].replace(" ", "\n", 1) for c in configs], fontsize=6.2)
    ax.set_ylabel("Mean latency per query (ms)")
    ax.legend(fontsize=6.8)
    ax.set_title(f"{DATASET_LABELS[dataset]}: latency decomposition", fontsize=8.5)
    return _save(fig, out_dir, "fig10_latency")


def fig_risk_coverage(rows: list[Row], out_dir: Path) -> Path:
    """Security-utility frontier: what each configuration buys, and what it costs.

    The x axis is clean accuracy (what benign users get), the y axis is one minus
    attack-attributable success (what the adversary does not get). A configuration in the
    top-right corner is strictly better than one below or to the left of it. Plotting the
    static and the adaptive adversary as separate series is the point of the figure: the
    configurations that dominate under static attack collapse to the bottom of the panel
    once the adversary is allowed to rewrite the payload.
    """
    import matplotlib.pyplot as plt

    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    static = [a for a in ATTACK_ORDER if a != "adaptive_poison"]
    clean = agg([r for r in rows if r["attack"] == "clean"], ["dataset", "config"],
                "clean_accuracy")
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.7), squeeze=False, sharey=True)
    for ax, dataset in zip(axes[0], datasets, strict=False):
        for attack_set, marker, label, edge in (
            (static, "o", "vs. static attacks", "none"),
            (["adaptive_poison"], "X", "vs. adaptive attack", "#111111"),
        ):
            pooled = seed_summary(
                [r for r in rows if r["dataset"] == dataset and r["attack"] in attack_set],
                ["config"], "conditional_asr", "n_baseline_correct",
            )
            for index, config in enumerate(CONFIG_ORDER):
                hit = pooled.get((config,))
                accuracy = clean.get((dataset, config))
                if hit is None or accuracy is None:
                    continue
                # Configurations frequently share a clean-accuracy value (a layer that only
                # filters at ingestion does not change benign answers at all), so nudge each
                # one by a fixed offset rather than letting identical points hide each other.
                accuracy = accuracy + (index - (len(CONFIG_ORDER) - 1) / 2) * 0.016
                ax.scatter(accuracy, 1.0 - hit[0], s=44, marker=marker,
                           color=CONFIG_COLORS[config], zorder=3,
                           edgecolor=edge, linewidth=0.7,
                           label=label if config == CONFIG_ORDER[0] else None)
        ax.set_xlim(-0.04, 1.06)
        ax.set_ylim(-0.06, 1.08)
        ax.set_xlabel("Clean accuracy")
        ax.set_title(DATASET_LABELS[dataset], fontsize=8.5)
    axes[0][0].set_ylabel("1 $-$ attributable ASR")

    # Two legends: marker encodes the adversary, colour encodes the configuration.
    from matplotlib.lines import Line2D

    adversary = [
        Line2D([], [], marker="o", linestyle="none", color=C_NEUTRAL, label="static attacks"),
        Line2D([], [], marker="X", linestyle="none", color=C_NEUTRAL,
               markeredgecolor="#111111", label="adaptive attack"),
    ]
    configs = [
        Line2D([], [], marker="s", linestyle="none", color=CONFIG_COLORS[c],
               label=CONFIG_LABELS[c])
        for c in CONFIG_ORDER
    ]
    first = fig.legend(handles=adversary, ncol=2, loc="lower center",
                       bbox_to_anchor=(0.5, -0.13), fontsize=6.8)
    fig.add_artist(first)
    fig.legend(handles=configs, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.31),
               fontsize=6.8)
    return _save(fig, out_dir, "fig11_frontier")


def fig_generator(rows: list[Row], out_dir: Path, dataset: str = "squad") -> Path:
    """How much residual risk survives when the generator refuses injected orders."""
    import matplotlib.pyplot as plt
    import numpy as np

    subset = [r for r in rows if r["dataset"] == dataset]
    # Attributable rates, so that the suppression attack is not credited with the pipeline's
    # own errors -- the comparison here is specifically about attack effectiveness.
    values = seed_summary(subset, ["config", "follow_injections", "attack"], "conditional_asr",
                       "n_baseline_correct")
    x = np.arange(len(ATTACK_ORDER))
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    series = [
        ("none", True, C_ATTACK, "", "Undefended, susceptible generator"),
        ("none", False, "#e79c93", "///", "Undefended, hardened generator"),
        ("full", True, C_DEFENSE, "", "Full defense, susceptible generator"),
        ("full", False, "#9dc6e0", "///", "Full defense, hardened generator"),
    ]
    width = 0.2
    for i, (config, follows, color, hatch, label) in enumerate(series):
        heights = [values.get((config, follows, a), (0.0,))[0] for a in ATTACK_ORDER]
        ax.bar(x + (i - 1.5) * width, heights, width, color=color, hatch=hatch,
               label=label, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS[a].replace(" ", "\n") for a in ATTACK_ORDER], fontsize=6.4)
    ax.set_ylabel("Attributable attack success rate")
    ax.set_ylim(0, 1.06)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.22),
               fontsize=6.6)
    return _save(fig, out_dir, "fig12_generator")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def tab_datasets(out_dir: Path) -> Path:
    """Corpus statistics for the three evaluation datasets."""
    from ragsec.eval.data import load_dataset
    from ragsec.eval.suite import DATASET_SOURCE, DATASETS

    body = []
    for key, spec in DATASETS.items():
        data = load_dataset(spec)
        lengths = [len(doc.text.split()) for doc in data.documents]
        body.append([
            DATASET_LABELS[key],
            str(len(data.documents)),
            str(len(data.qa)),
            f"{statistics.fmean(lengths):.0f}",
            f"{len(data.documents) / max(1, len(data.qa)):.1f}",
            DATASET_SOURCE[key].replace("_", "\\_"),
        ])
    return _write(
        out_dir / "tab_datasets.tex",
        latex_table(
            "Evaluation corpora. The document-to-question ratio indicates how much "
            "topically adjacent but non-answering material a retriever must sort through; "
            "HotpotQA contributes its distractor paragraphs, which is why its ratio is high.",
            "tab:datasets",
            ["Dataset", "Documents", "Questions", "Mean words/doc", "Docs/question", "Source id"],
            body,
            column_spec="lrrrrl",
        ),
    )


def tab_main(rows: list[Row], out_dir: Path) -> Path:
    """ASR for every configuration and attack, raw and attack-attributable."""
    values = seed_summary(rows, ["dataset", "config", "attack"])
    attributable = seed_summary(rows, ["dataset", "config", "attack"], "conditional_asr",
                             "n_baseline_correct")
    clean = agg([r for r in rows if r["attack"] == "clean"], ["dataset", "config"],
                "clean_accuracy")
    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    body: list[list[str]] = []
    for dataset in datasets:
        body.append([f"\\multicolumn{{7}}{{l}}{{\\textit{{{DATASET_LABELS[dataset]}}}}}"])
        for config in CONFIG_ORDER:
            cells = [f"\\quad {CONFIG_LABELS[config]}"]
            for attack in ATTACK_ORDER:
                hit = values.get((dataset, config, attack))
                cond = attributable.get((dataset, config, attack))
                cells.append(
                    f"{hit[0]:.2f} \\tiny({cond[0]:.2f})" if hit and cond
                    else f"{hit[0]:.2f}" if hit
                    else "n/a"
                )
            cells.append(fmt(clean.get((dataset, config))))
            body.append(cells)
    return _write(
        out_dir / "tab_main.tex",
        latex_table(
            "Attack success rate by defense configuration, pooled over query-seed trials. "
            "Each cell reads \\emph{raw} (\\emph{attack-attributable}), the latter "
            "restricted to queries the same configuration answered correctly with no adversary "
            "present. The two diverge only for the suppression attack, whose raw rate absorbs "
            "the pipeline's own error rate; this is most visible on HotpotQA, where multi-hop "
            "questions depress clean accuracy. Figure~\\ref{fig:bars} shows the observed "
            "minimum-to-maximum range across seeds. The last column is clean accuracy on the "
            "unattacked corpus, i.e.\\ the utility cost of the configuration.",
            "tab:main",
            ["Configuration", *[ATTACK_LABELS[a] for a in ATTACK_ORDER], "Clean acc."],
            body,
            column_spec="l" + "c" * len(ATTACK_ORDER) + "c",
        ),
    )


def tab_full(rows: list[Row], out_dir: Path, dataset: str = "squad") -> Path:
    """Detailed behaviour of the two strongest configurations."""
    subset = [r for r in rows if r["dataset"] == dataset]
    values = seed_summary(subset, ["config", "attack"])
    conditional = seed_summary(
        subset, ["config", "attack"], "conditional_asr", "n_baseline_correct"
    )
    body: list[list[str]] = []
    for config in ("full", "full_signed"):
        body.append([f"\\multicolumn{{9}}{{l}}{{\\textit{{{CONFIG_LABELS[config]}}}}}"])
        for attack in [*ATTACK_ORDER, "clean"]:
            cells_rows = [r for r in subset if r["config"] == config and r["attack"] == attack]
            if not cells_rows:
                continue
            hit = values.get((config, attack))
            cond = conditional.get((config, attack))
            accuracy = agg(cells_rows, ["attack"],
                           "clean_accuracy" if attack == "clean" else "attacked_accuracy")
            body.append([
                f"\\quad {ATTACK_LABELS[attack]}",
                f"{hit[0]:.2f}" if hit else "n/a",
                f"{cond[0]:.2f}" if cond else "n/a",
                fmt(accuracy.get((attack,))),
                fmt(agg(cells_rows, ["attack"], "poison_retrieval_rate").get((attack,))),
                fmt(agg(cells_rows, ["attack"], "poison_used_rate").get((attack,))),
                fmt(agg(cells_rows, ["attack"], "filter_tpr").get((attack,))),
                fmt(agg(cells_rows, ["attack"], "filter_fpr").get((attack,))),
                fmt(agg(cells_rows, ["attack"], "abstention_rate").get((attack,))),
            ])
    return _write(
        out_dir / "tab_full.tex",
        latex_table(
            f"Detailed behaviour of the layered defense on {DATASET_LABELS[dataset]}. "
            "\\emph{Attributable ASR} restricts attack success to the queries this same "
            "configuration answered correctly with no adversary present; the distinction "
            "matters for the suppression attack, since a pipeline that was already going to "
            "fail cannot meaningfully be said to have been suppressed. \\emph{Poison "
            "retrieved} is the fraction of attacked queries for which a poisoned chunk "
            "entered the candidate set; \\emph{poison used} is the fraction for which one "
            "supported the answer that was returned.",
            "tab:full",
            ["Attack", "ASR", "Attrib. ASR", "Accuracy", "Poison retr.", "Poison used",
             "Filter TPR", "Filter FPR", "Abstention"],
            body,
            column_spec="lrrrrrrrr",
        ),
    )


def tab_evasion(rows: list[Row], out_dir: Path, dataset: str = "squad") -> Path:
    """Which evasion technique the adaptive attacker actually needs."""
    subset = [r for r in rows if r["dataset"] == dataset]
    values = seed_summary(subset, ["config", "disabled"])
    tpr = agg(subset, ["config", "disabled"], "filter_tpr")
    retrieved = agg(subset, ["config", "disabled"], "poison_retrieval_rate")
    accuracy = agg(subset, ["config", "disabled"], "attacked_accuracy")
    body: list[list[str]] = []
    for key in ["none", *EVASION_LABELS]:
        label = ("All six techniques enabled" if key == "none"
                 else f"$-$ {EVASION_LABELS[key]}")
        full = values.get(("full", key))
        signed = values.get(("full_signed", key))
        body.append([
            label,
            f"{full[0]:.2f} \\tiny[{full[1]:.2f},{full[2]:.2f}]" if full else "n/a",
            fmt(retrieved.get(("full", key))),
            fmt(tpr.get(("full", key))),
            fmt(accuracy.get(("full", key))),
            f"{signed[0]:.2f}" if signed else "n/a",
        ])
    return _write(
        out_dir / "tab_evasion.tex",
        latex_table(
            "Leave-one-out ablation of the adaptive attacker's evasion techniques on "
            f"{DATASET_LABELS[dataset]} ($3$ poisoned documents per target). Each row disables "
            "one technique and leaves the other five in place, so a drop in ASR identifies a "
            "technique the defense depends on the adversary \\emph{not} using. Bracketed "
            "values give the observed min-to-max range across seeds, not a confidence interval.",
            "tab:evasion",
            ["Adaptive attacker variant", "ASR (full)", "Poison retr.", "Filter TPR",
             "Accuracy", "ASR (signed)"],
            body,
            column_spec="lrrrrr",
        ),
    )


def tab_provenance(rows: list[Row], out_dir: Path) -> Path:
    """Ingestion controls of increasing strength."""
    order = ["open", "trust_only", "allowlist", "quota", "signed"]
    labels = {
        "open": "No ingestion control",
        "trust_only": "Trust threshold ($\\theta_{\\mathrm{trust}}=0.6$)",
        "allowlist": "Source allowlist $+$ trust threshold",
        "quota": "Allowlist $+$ per-source quota",
        "signed": "HMAC-signed ingestion (allowlist $+$ signature)",
    }
    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    values = seed_summary(rows, ["dataset", "variant", "attack"])
    rejected = agg(rows, ["dataset", "variant", "attack"], "poison_docs_rejected")
    body = []
    for variant in order:
        cells = [labels[variant]]
        for dataset in datasets:
            for attack in ("poisoned_rag", "adaptive_poison"):
                hit = values.get((dataset, variant, attack))
                cells.append(f"{hit[0]:.2f}" if hit else "n/a")
        cells.append(fmt(rejected.get((datasets[0], variant, "adaptive_poison")), 0))
        body.append(cells)
    header = ["Ingestion control"]
    for dataset in datasets:
        header += [f"{DATASET_LABELS[dataset]} static", f"{DATASET_LABELS[dataset]} adaptive"]
    header.append("Rejected docs")
    return _write(
        out_dir / "tab_provenance.tex",
        latex_table(
            "Attack success rate under ingestion controls of increasing strength, against a "
            "static (PoisonedRAG) and a provenance-spoofing adaptive adversary, $3$ documents "
            "per target. The final column counts poisoned documents rejected at ingestion for "
            "the adaptive adversary on the first dataset.",
            "tab:provenance",
            header,
            body,
            column_spec="l" + "r" * (len(header) - 1),
        ),
    )


def tab_detection(rows: list[Row], threshold_rows: list[Row], out_dir: Path) -> Path:
    """Threshold-free detection quality plus the deployed operating point.

    All three quantities are read from the retrieval-filter-only configuration. In the full
    stack, Layer 1 rejects the static attacks before they are indexed, so the filter never sees
    a poisoned candidate and its detection scores there are undefined rather than perfect --
    reporting them from the full stack would credit Layer 2 with Layer 1's work.
    """
    l2 = [r for r in rows if r["config"] == "L2"]
    auroc = agg(l2, ["dataset", "attack"], "detection_auroc")
    auprc = agg(l2, ["dataset", "attack"], "detection_auprc")
    tpr = agg(l2, ["dataset", "attack"], "filter_tpr")
    fpr = agg(l2, ["dataset", "attack"], "filter_fpr")
    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    body = []
    for attack in ATTACK_ORDER:
        cells = [ATTACK_LABELS[attack]]
        for dataset in datasets:
            cells += [
                fmt(auroc.get((dataset, attack))),
                fmt(auprc.get((dataset, attack))),
                fmt(tpr.get((dataset, attack))),
            ]
        body.append(cells)
    clean_fpr = [fmt(fpr.get((d, "clean"))) for d in datasets]
    header = ["Attack"]
    for dataset in datasets:
        header += [f"{DATASET_LABELS[dataset]} AUROC", "AUPRC", "TPR"]
    return _write(
        out_dir / "tab_detection.tex",
        latex_table(
            "Retrieval-filter detection quality, measured in the filter-only configuration so "
            "that Layer 1 does not remove the candidates the filter is being scored on. AUROC "
            "and average precision (AUPRC) are threshold-free and computed over every retrieval "
            "candidate; TPR is measured at the deployed operating point. Poisoned candidates "
            "are a small minority of the candidate set, so AUPRC is the more informative "
            "summary. False-positive rate on "
            "unattacked traffic was " + ", ".join(
                f"{v} ({DATASET_LABELS[d]})" for v, d in zip(clean_fpr, datasets, strict=False)
            ) + ".",
            "tab:detection",
            header,
            body,
            column_spec="l" + "rrr" * len(datasets),
        ),
    )


def tab_latency(rows: list[Row], out_dir: Path, dataset: str = "squad") -> Path:
    """Latency decomposition and its relation to the candidate multiplier."""
    subset = [r for r in rows if r["dataset"] == dataset and r["attack"] != "clean"]
    body = []
    for config in CONFIG_ORDER:
        cells_rows = [r for r in subset if r["config"] == config]
        if not cells_rows:
            continue
        body.append([
            CONFIG_LABELS[config],
            fmt(agg(cells_rows, ["config"], "mean_retrieve_ms").get((config,)), 2),
            fmt(agg(cells_rows, ["config"], "mean_filter_ms").get((config,)), 2),
            fmt(agg(cells_rows, ["config"], "mean_generate_ms").get((config,)), 2),
            fmt(agg(cells_rows, ["config"], "mean_latency_ms").get((config,)), 2),
            fmt(agg(cells_rows, ["config"], "p95_latency_ms").get((config,)), 2),
        ])
    return _write(
        out_dir / "tab_latency.tex",
        latex_table(
            f"Per-query latency on {DATASET_LABELS[dataset]}, averaged over all attack runs. "
            "Times exclude external model and network cost: the generator is a local mock, so "
            "these numbers isolate the overhead the defense layers themselves add.",
            "tab:latency",
            ["Configuration", "Retrieval (ms)", "Filtering (ms)", "Generation (ms)",
             "Mean (ms)", "p95 (ms)"],
            body,
            column_spec="lrrrrr",
        ),
    )


def tab_generator(rows: list[Row], out_dir: Path, dataset: str = "squad") -> Path:
    """Attack success with a susceptible and with a hardened generator."""
    subset = [r for r in rows if r["dataset"] == dataset]
    values = seed_summary(subset, ["config", "follow_injections", "attack"], "conditional_asr",
                       "n_baseline_correct")
    body = []
    for attack in ATTACK_ORDER:
        cells = [ATTACK_LABELS[attack]]
        for config in ("none", "full"):
            for follows in (True, False):
                hit = values.get((config, follows, attack))
                cells.append(f"{hit[0]:.2f}" if hit else "n/a")
        body.append(cells)
    return _write(
        out_dir / "tab_generator.tex",
        latex_table(
            "Contribution of generator susceptibility, as attack-attributable success rate. "
            "A \\emph{hardened} generator refuses "
            "instructions found in retrieved text but still answers extractively from it, so "
            "the residual risk in that column is attributable to retrieval rather than to "
            "instruction following.",
            "tab:generator",
            ["Attack", "Undefended, susceptible", "Undefended, hardened",
             "Defended, susceptible", "Defended, hardened"],
            body,
            column_spec="lrrrr",
        ),
    )


def tab_utility(rows: list[Row], out_dir: Path) -> Path:
    """The cost each configuration imposes on benign traffic."""
    clean = [r for r in rows if r["attack"] == "clean"]
    accuracy = agg(clean, ["dataset", "config"], "clean_accuracy")
    abstention = agg(clean, ["dataset", "config"], "abstention_rate")
    fpr = agg(clean, ["dataset", "config"], "filter_fpr")
    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in clean)]
    body = []
    for config in CONFIG_ORDER:
        cells = [CONFIG_LABELS[config]]
        for dataset in datasets:
            cells += [
                fmt(accuracy.get((dataset, config))),
                fmt(abstention.get((dataset, config))),
                fmt(fpr.get((dataset, config))),
            ]
        body.append(cells)
    header = ["Configuration"]
    for dataset in datasets:
        header += [f"{DATASET_LABELS[dataset]} acc.", "Abstain", "FPR"]
    return _write(
        out_dir / "tab_utility.tex",
        latex_table(
            "Utility on unattacked traffic. Accuracy is measured against the undefended "
            "pipeline's own ceiling; abstention is the rate at which the aggregation layer "
            "declines to answer, and FPR the rate at which the retrieval filter flags "
            "legitimate evidence.",
            "tab:utility",
            header,
            body,
            column_spec="l" + "rrr" * len(datasets),
        ),
    )


def tab_examples(out_dir: Path) -> Path:
    """One crafted passage per attack, so the constructions can be compared directly."""
    from ragsec.attacks import ATTACK_REGISTRY, AttackTarget, get_attack
    from ragsec.core.models import Chunk, Provenance, RetrievedChunk
    from ragsec.defenses.retrieval import FilterContext, default_filters

    query = "Which NFL team represented the AFC at Super Bowl 50?"
    target = AttackTarget(query, "Carolina Panthers")
    ctx = FilterContext.build(query)
    filters = default_filters()

    def escape(text: str) -> str:
        for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                     ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"),
                     ("}", "\\}"), ("~", "\\textasciitilde{}"),
                     ("\u200b", "[ZWSP]")):
            text = text.replace(a, b)
        return text

    body = []
    for name in ATTACK_ORDER:
        if name not in ATTACK_REGISTRY:
            continue
        attack = get_attack(name)
        attack.bind_corpus("huggingface")
        text = attack.craft(target, 0)
        chunk = Chunk(text=text, doc_id="ex", chunk_id="ex:0", position=0,
                      provenance=Provenance(source_id=attack.source_for(0)[0]))
        rc = RetrievedChunk(chunk=chunk, score=0.6, rank=0, index=0)
        flagged = [f.name for f in filters if f.evaluate(rc, ctx).flagged]
        shown = text if len(text) <= 240 else text[:237] + "..."
        body.append([
            ATTACK_LABELS[name],
            f"\\texttt{{\\footnotesize {escape(shown)}}}",
            escape(attack.source_for(0)[0]),
            ", ".join(f.replace("_", " ") for f in flagged) or "\\emph{none}",
        ])
    return _write(
        out_dir / "tab_examples.tex",
        latex_table(
            "One poisoned passage per attack for the query \\emph{``Which NFL team represented "
            "the AFC at Super Bowl 50?''} with decoy answer \\emph{Carolina Panthers}, together "
            "with the source the passage claims and the retrieval filters it trips. The static "
            "constructions each carry a syntactic tell; the adaptive passage carries none, and "
            "claims the corpus's own source identifier.",
            "tab:examples",
            ["Attack", "Crafted passage", "Claimed source", "Filters triggered"],
            body,
            column_spec="lXll",
            wide=True,
        ),
    )


# ---------------------------------------------------------------------------
# Inline macros
#
# Numbers quoted in the prose are emitted as LaTeX macros, so the text cannot drift out of
# sync with the results the way hand-copied figures do.
# ---------------------------------------------------------------------------


def macros(studies: dict[str, list[Row]], out_dir: Path) -> Path:
    main = studies.get("main", [])
    evasion = studies.get("evasion", [])
    provenance = studies.get("provenance", [])
    lines: list[str] = [
        "% Generated by ragsec.eval.report - do not edit by hand.",
    ]

    def macro(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\{macro_name(name)}}}{{{value}}}")

    values = seed_summary(main, ["dataset", "config", "attack"])
    attributable = seed_summary(main, ["dataset", "config", "attack"], "conditional_asr",
                             "n_baseline_correct")
    static = [a for a in ATTACK_ORDER if a != "adaptive_poison"]
    for dataset in {r["dataset"] for r in main}:
        tag = dataset.capitalize()
        for config in CONFIG_ORDER:
            suffix = f"{tag}{config.replace('_', '')}"
            static_rows = [
                r for r in main
                if r["dataset"] == dataset and r["config"] == config and r["attack"] in static
            ]
            # Both the raw and the attack-attributable pooled rates: the suppression attack
            # inflates the former, so the prose needs to be able to quote either.
            for prefix, kwargs in (
                ("asrStatic", {}),
                ("casrStatic",
                 {"rate_field": "conditional_asr", "n_field": "n_baseline_correct"}),
            ):
                pooled = seed_summary(static_rows, [], **kwargs)
                if pooled.get(()):
                    macro(f"{prefix}{suffix}", f"{pooled[()][0]:.2f}")
            hit = values.get((dataset, config, "adaptive_poison"))
            if hit:
                macro(f"asrAdaptive{suffix}", f"{hit[0]:.2f}")
            hit = attributable.get((dataset, config, "adaptive_poison"))
            if hit:
                macro(f"casrAdaptive{suffix}", f"{hit[0]:.2f}")
            hit = attributable.get((dataset, config, "context_flooding"))
            if hit:
                macro(f"casrFlooding{suffix}", f"{hit[0]:.2f}")
            hit = values.get((dataset, config, "context_flooding"))
            if hit:
                macro(f"asrFlooding{suffix}", f"{hit[0]:.2f}")
        clean = agg([r for r in main if r["dataset"] == dataset and r["attack"] == "clean"],
                    ["config"], "clean_accuracy")
        for config in CONFIG_ORDER:
            if (config,) in clean:
                macro(f"cleanAcc{tag.capitalize()}{config.replace('_', '')}",
                      f"{clean[(config,)]:.2f}")

    if evasion:
        ev = seed_summary([r for r in evasion if r["dataset"] == "squad"], ["config", "disabled"])
        for key in ["none", *EVASION_LABELS]:
            for config in ("full", "full_signed"):
                hit = ev.get((config, key))
                if hit:
                    name = key.replace("_", "").capitalize()
                    macro(f"evasion{config.replace('_', '').capitalize()}{name}",
                          f"{hit[0]:.2f}")
    if provenance:
        pv = seed_summary([r for r in provenance if r["dataset"] == "squad"],
                       ["variant", "attack"])
        # The quota rung behaves differently per corpus (it is a volume control, so what
        # matters is the campaign-to-corpus ratio), so the prose quotes a second dataset.
        hotpot = seed_summary([r for r in provenance if r["dataset"] == "hotpotqa"],
                           ["variant", "attack"])
        for variant in ("quota", "signed"):
            hit = hotpot.get((variant, "adaptive_poison"))
            if hit:
                macro(f"prov{variant.capitalize()}AdaptiveHotpot", f"{hit[0]:.2f}")
        for variant in ("open", "trust_only", "allowlist", "quota", "signed"):
            for attack in ("poisoned_rag", "adaptive_poison"):
                hit = pv.get((variant, attack))
                if hit:
                    macro(
                        f"prov{variant.replace('_', '').capitalize()}"
                        f"{'Static' if attack == 'poisoned_rag' else 'Adaptive'}",
                        f"{hit[0]:.2f}",
                    )
    # Detection quality is read from the filter-only configuration, for the reason given in
    # tab_detection: in the full stack Layer 1 removes the candidates being scored.
    l2 = [r for r in main if r["config"] == "L2"]
    static = [a for a in ATTACK_ORDER if a != "adaptive_poison"]
    for field, prefix in (("detection_auroc", "auroc"), ("detection_auprc", "auprc")):
        by_attack = agg(l2, ["dataset", "attack"], field)
        for dataset in {r["dataset"] for r in main}:
            tag = dataset.capitalize()
            adaptive = by_attack.get((dataset, "adaptive_poison"))
            if adaptive is not None:
                macro(f"{prefix}Adaptive{tag}", f"{adaptive:.2f}")
            values = [
                by_attack[(dataset, a)] for a in static if (dataset, a) in by_attack
            ]
            if values:
                macro(f"{prefix}StaticMin{tag}", f"{min(values):.2f}")
    tpr = agg(l2, ["dataset", "attack"], "filter_tpr")
    for dataset in {r["dataset"] for r in main}:
        hit = tpr.get((dataset, "adaptive_poison"))
        if hit is not None:
            macro(f"tprAdaptive{dataset.capitalize()}", f"{hit:.2f}")
    # Two representative operating points from the threshold sweep, so the prose can state
    # what buying detection of the adaptive attack actually costs.
    threshold_rows = studies.get("threshold", [])
    if threshold_rows:
        squad = [r for r in threshold_rows if r["dataset"] == "squad"]
        adaptive = agg([r for r in squad if r["attack"] == "adaptive_poison"],
                       ["threshold"], "filter_tpr")
        adaptive_asr = seed_summary([r for r in squad if r["attack"] == "adaptive_poison"],
                                 ["threshold"], "conditional_asr", "n_baseline_correct")
        clean = agg([r for r in squad if r["attack"] == "clean"], ["threshold"],
                    "clean_accuracy")
        fpr = agg([r for r in squad if r["attack"] == "clean"], ["threshold"], "filter_fpr")
        for tau, name in ((1.0, "Default"), (0.7, "Moderate"), (0.5, "Aggressive")):
            macro(f"tau{name}", f"{tau:.1f}")
            if (tau,) in adaptive:
                macro(f"tau{name}TprAdaptive", f"{adaptive[(tau,)]:.2f}")
            hit = adaptive_asr.get((tau,))
            if hit:
                macro(f"tau{name}AsrAdaptive", f"{hit[0]:.2f}")
            if (tau,) in clean:
                macro(f"tau{name}CleanAcc", f"{clean[(tau,)]:.2f}")
            if (tau,) in fpr:
                macro(f"tau{name}Fpr", f"{fpr[(tau,)]:.2f}")

    # Adversary budget: the interesting quantity is how many corroborating documents the
    # adaptive attacker needs before the aggregation layer's redundancy bonus flips in their
    # favour, so the extremes and the b=1/b=2 step are both emitted.
    budget = studies.get("budget", [])
    if budget:
        for dataset in {r["dataset"] for r in budget}:
            tag = dataset.capitalize()
            for attack, aname in (("adaptive_poison", "Adaptive"), ("poisoned_rag", "Static")):
                for config, cname in (("none", "None"), ("full", "Full"),
                                      ("full_signed", "Signed")):
                    pooled = seed_summary(
                        [r for r in budget if r["dataset"] == dataset
                         and r["config"] == config and r["attack"] == attack],
                        ["n_per_target"], "conditional_asr", "n_baseline_correct",
                    )
                    for n in (1, 2, 20):
                        hit = pooled.get((n,))
                        if hit:
                            macro(f"budget{aname}{cname}{tag}B{n}", f"{hit[0]:.2f}")

    # Over-retrieval sweep at both operating points, at the extremes of the multiplier range.
    static_attacks = [a for a in ATTACK_ORDER if a != "adaptive_poison"]
    for study, tag in (("backfill", "Default"), ("backfill_strict", "Strict")):
        source = [r for r in studies.get(study, []) if r["dataset"] == "squad"]
        if not source:
            continue
        clean = [r for r in source if r["attack"] == "clean"]
        acc = agg(clean, ["candidate_multiplier"], "clean_accuracy")
        abstain = agg(clean, ["candidate_multiplier"], "abstention_rate")
        fpr = agg(clean, ["candidate_multiplier"], "filter_fpr")
        latency = agg(source, ["candidate_multiplier"], "p95_latency_ms")
        asr = seed_summary([r for r in source if r["attack"] in static_attacks],
                        ["candidate_multiplier"], "conditional_asr", "n_baseline_correct")
        for m, edge in ((1, "Lo"), (20, "Hi")):
            if (m,) in acc:
                macro(f"backfill{tag}Acc{edge}", f"{acc[(m,)]:.2f}")
            if (m,) in abstain:
                macro(f"backfill{tag}Abstain{edge}", f"{abstain[(m,)]:.2f}")
            if (m,) in fpr:
                macro(f"backfill{tag}Fpr{edge}", f"{fpr[(m,)]:.2f}")
            if (m,) in latency:
                macro(f"backfill{tag}Lat{edge}", f"{latency[(m,)]:.0f}")
            hit = asr.get((m,))
            if hit:
                macro(f"backfill{tag}Asr{edge}", f"{hit[0]:.2f}")

    # Latency figures quoted in the prose.
    attacked = [r for r in main if r["attack"] != "clean"]
    for field, name in (("mean_filter_ms", "filterMs"), ("mean_latency_ms", "meanMs"),
                        ("p95_latency_ms", "pMs")):
        by_config = agg(attacked, ["dataset", "config"], field)
        for (dataset, config), value in by_config.items():
            tag = f"{config.replace('_', '').capitalize()}{dataset.capitalize()}"
            macro(f"{name}{tag}", f"{value:.0f}" if value >= 10 else f"{value:.1f}")
    macro("nSeeds", str(len({r["seed"] for r in main}) or 3))
    return _write(out_dir / "macros.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


#: Macros the manuscript body may use that are not generated here (LaTeX or class built-ins,
#: plus the commands defined in the drivers' preambles).
_KNOWN_TEX_MACROS = frozenset(
    """
    begin end section subsection subsubsection label ref citep citet cite input includegraphics
    textwidth textbf textit texttt emph item itemize enumerate description figure table caption
    toprule midrule bottomrule multicolumn tabularx tabular quad qquad footnotesize small
    noindent hfill vspace paperdir algorithm algorithmic Require Ensure State If Else ElsIf EndIf
    For ForAll EndFor While EndWhile Return Comment Procedure EndProcedure textsc frac max min
    sum cdot ge le neq in notin gets bot mathrm mathcal sum_ dots ldots times approx equation
    clip sigma tau rho gamma beta mu theta pi delta kappa Sigma phi lambda alpha epsilon
    top_k top_1 begin_ minipage centering hline midrule_ providecommand newcommand
    """.split()
)


def verify_macros(body: Path, macros: Path) -> list[str]:
    """Return macros the body uses that ``macros.tex`` does not define.

    The manuscript quotes result numbers through generated macros so the prose cannot drift out
    of sync with the data. That guarantee is only worth anything if a typo in a macro name is
    caught here rather than appearing in the PDF as a silent undefined-control-sequence error.
    """
    import re

    if not body.exists() or not macros.exists():
        return []
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros.read_text(encoding="utf-8")))
    # Match greedily over letters *and* digits so that an illegal name such as
    # "casrStaticSquadL3" is reported rather than silently truncated to a legal prefix.
    used = set(re.findall(r"\\([A-Za-z][A-Za-z0-9]*)", body.read_text(encoding="utf-8")))
    # Result macros are the ones this module emits; anything else is LaTeX's business.
    prefixes = ("asr", "casr", "cleanAcc", "evasion", "prov", "nSeeds")
    candidates = {m for m in used if m.startswith(prefixes)}
    return sorted(candidates - defined - _KNOWN_TEX_MACROS)


def build_report(results_dir: Path, output_dir: Path) -> list[Path]:
    """Regenerate every table and figure the manuscript depends on."""
    _style()
    studies = {name: load_rows(results_dir, name) for name in
               ("main", "evasion", "budget", "threshold", "backfill", "backfill_strict",
                "provenance", "generator")}
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    written: list[Path] = [
        fig_architecture(figures), tab_datasets(tables), tab_examples(tables)
    ]

    def attempt(label: str, fn: Callable[[], Path]) -> None:
        try:
            written.append(fn())
        except Exception as exc:  # noqa: BLE001 - a partial suite should still report
            print(f"  skipped {label}: {type(exc).__name__}: {exc}")

    if studies["main"]:
        attempt("tab_main", lambda: tab_main(studies["main"], tables))
        attempt("tab_full", lambda: tab_full(studies["main"], tables))
        attempt("tab_utility", lambda: tab_utility(studies["main"], tables))
        attempt("tab_latency", lambda: tab_latency(studies["main"], tables))
        attempt("tab_detection",
                lambda: tab_detection(studies["main"], studies["threshold"], tables))
        attempt("fig_asr_heatmap", lambda: fig_asr_heatmap(studies["main"], figures))
        attempt("fig_asr_bars", lambda: fig_asr_bars(studies["main"], figures))
        attempt("fig_latency", lambda: fig_latency(studies["main"], figures))
        attempt("fig_risk_coverage", lambda: fig_risk_coverage(studies["main"], figures))
    if studies["evasion"]:
        attempt("tab_evasion", lambda: tab_evasion(studies["evasion"], tables))
        attempt("fig_evasion", lambda: fig_evasion(studies["evasion"], figures))
    if studies["budget"]:
        attempt("fig_budget", lambda: fig_budget(studies["budget"], figures))
    if studies["threshold"]:
        attempt("fig_detection", lambda: fig_detection(studies["threshold"], figures))
        attempt("fig_threshold", lambda: fig_threshold(studies["threshold"], figures))
    if studies["backfill"]:
        attempt(
            "fig_backfill",
            lambda: fig_backfill(studies["backfill"], figures, studies["backfill_strict"]),
        )
    if studies["provenance"]:
        attempt("tab_provenance", lambda: tab_provenance(studies["provenance"], tables))
        attempt("fig_provenance", lambda: fig_provenance(studies["provenance"], figures))
    if studies["generator"]:
        attempt("tab_generator", lambda: tab_generator(studies["generator"], tables))
        attempt("fig_generator", lambda: fig_generator(studies["generator"], figures))
    written.append(macros(studies, tables))
    missing = verify_macros(output_dir / "body.tex", tables / "macros.tex")
    if missing:
        print(
            "  WARNING: the manuscript quotes macros that no study has produced yet "
            f"({len(missing)}): {', '.join(missing)}"
        )
    return written
