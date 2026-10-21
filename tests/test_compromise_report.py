"""Tests for the compromised-publisher study and its reporting.

These cover the plumbing that turns the study's rows into the manuscript's numbers: that the
studies are wired into the suite, that abstention is pooled on the attacked denominator rather
than the clean one, and that the generated table reports the two corroboration units against
each other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragsec.eval.report import macro_name, macros, tab_compromise
from ragsec.eval.suite import (
    DEFAULT_PUBLISHERS,
    DIVERSITY_BUDGET,
    STUDIES,
    study_compromise,
    study_diversity,
)


def _row(
    config: str,
    attack: str,
    n_per_target: int,
    seed: int,
    *,
    conditional_asr: float | None = None,
    abstention_rate: float = 0.0,
    clean_accuracy: float | None = None,
    dataset: str = "squad",
) -> dict:
    return {
        "study": "compromise",
        "dataset": dataset,
        "config": config,
        "attack": attack,
        "n_per_target": n_per_target,
        "seed": seed,
        "conditional_asr": conditional_asr,
        "n_baseline_correct": 100 if conditional_asr is not None else 0,
        "abstention_rate": abstention_rate,
        "clean_accuracy": clean_accuracy,
        "n_attacked": 200 if attack != "clean" else 0,
        "n_clean": 0 if attack != "clean" else 200,
        "attack_success_rate": conditional_asr or 0.0,
    }


def _study_rows() -> list[dict]:
    rows: list[dict] = []
    for config, casr, abstain in (
        ("full_signed", 0.80, 0.08),
        ("full_signed_independent", 0.60, 0.28),
    ):
        for seed in (0, 1, 2):
            rows.append(
                _row(config, "adaptive_poison", 5, seed,
                     conditional_asr=casr, abstention_rate=abstain)
            )
            rows.append(
                _row(config, "clean", 5, seed,
                     abstention_rate=0.27, clean_accuracy=0.51)
            )
    return rows


# ---------------------------------------------------------------------------
# Suite wiring
# ---------------------------------------------------------------------------


def test_both_studies_are_registered() -> None:
    assert "compromise" in STUDIES
    assert "diversity" in STUDIES


def test_compromise_cells_model_a_compromised_publisher() -> None:
    cells = list(study_compromise(["squad"], [0]))

    assert cells, "expected at least one cell"
    for cell in cells:
        assert cell.config.sign_corpus is True
        assert cell.config.compromised_publisher is True
        assert cell.config.publishers == DEFAULT_PUBLISHERS
        assert cell.config.defenses, "the stack must be present"


def test_compromise_compares_the_two_corroboration_units() -> None:
    configs = {cell.key["config"] for cell in study_compromise(["squad"], [0])}

    assert configs == {"full_signed", "full_signed_independent"}


def test_diversity_varies_publishers_at_a_fixed_budget() -> None:
    cells = list(study_diversity(["squad"], [0]))

    publishers = {cell.config.publishers for cell in cells}
    budgets = {
        attack["n_per_target"] for cell in cells for attack in cell.config.attacks
    }

    assert len(publishers) > 1, "the control must vary publisher diversity"
    assert budgets == {DIVERSITY_BUDGET}, "the budget must be held fixed"
    assert all(cell.config.compromised_publisher for cell in cells)


def test_diversity_includes_the_single_publisher_case() -> None:
    """With one publisher there is no honest diversity, which is the informative extreme."""
    assert 1 in {cell.config.publishers for cell in study_diversity(["squad"], [0])}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_table_reports_both_rules_with_denominator(tmp_path: Path) -> None:
    path = tab_compromise(_study_rows(), tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "Document identifier" in text
    assert "Verified signer" in text
    assert "0.80" in text
    assert "0.60" in text
    # The clean-correct denominator must be reported, pooled over the three seeds.
    assert "300" in text
    assert r"\label{tab:compromise}" in text


def test_table_carries_intervals_when_they_are_supplied(tmp_path: Path) -> None:
    uncertainty = [{
        "dataset": "squad",
        "config": "full_signed",
        "attack": "adaptive_poison",
        "n_per_target": 5,
        "conditional_asr_ci_low": 0.74,
        "conditional_asr_ci_high": 0.86,
    }]

    text = tab_compromise(_study_rows(), tmp_path, uncertainty).read_text(encoding="utf-8")

    assert "0.80 [0.74, 0.86]" in text


def test_macros_emit_both_units_and_the_abstention_cost(tmp_path: Path) -> None:
    path = macros({"compromise": _study_rows()}, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert r"\newcommand{\compromiseSignedSquadBFive}{0.80}" in text
    assert r"\newcommand{\compromiseSignedIndependentSquadBFive}{0.60}" in text
    assert r"\newcommand{\compromiseAbstainSignedSquadBFive}{0.08}" in text
    assert r"\newcommand{\compromiseAbstainSignedIndependentSquadBFive}{0.28}" in text


def test_clean_accuracy_macro_uses_the_clean_denominator(tmp_path: Path) -> None:
    """Clean rows carry n_attacked = 0, so pooling them on that denominator loses them."""
    text = macros({"compromise": _study_rows()}, tmp_path).read_text(encoding="utf-8")

    assert r"\newcommand{\compromiseCleanAccSignedSquadBFive}{0.51}" in text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("compromiseSignedSquadB5", "compromiseSignedSquadBFive"),
        ("diversitySignedIndependentSquadP16", "diversitySignedIndependentSquadPOneSix"),
    ],
)
def test_generated_macro_names_are_legal_tex(raw: str, expected: str) -> None:
    name = macro_name(raw)

    assert name == expected
    assert name.isalpha(), "a TeX control word may contain letters only"


# ---------------------------------------------------------------------------
# Confirmation-study selection
# ---------------------------------------------------------------------------


def test_the_new_configurations_are_selectable_for_confirmation() -> None:
    """Regression: the default confirmation grid is not the set of permitted choices.

    Validating a caller's selection against CONFIRMATION_CONFIGS rejected
    full_independent, which defense_stack resolves perfectly well, so the real-model
    confirmation of the mechanism could not be requested at all.
    """
    from ragsec.eval.suite import CONFIG_LABELS, defense_stack

    for config in ("full_independent", "full_signed_independent", "L3_independent"):
        assert config in CONFIG_LABELS
        assert defense_stack(config, "squad"), f"{config} must resolve to a stack"


def test_an_unknown_confirmation_configuration_is_still_rejected() -> None:
    from ragsec.eval.suite import run_suite

    with pytest.raises(ValueError, match="Unknown confirmation configurations"):
        run_suite(
            studies=["confirmation"],
            confirmation_configs=["not_a_configuration"],
        )
