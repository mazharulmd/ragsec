"""Static checks on the manuscript sources.

LaTeX reports an undefined reference or a missing citation as a warning buried in a log, and a
missing figure only when the file is compiled. This module catches those without a TeX
installation, so that `ragsec figures` fails loudly when the generated artefacts and the
manuscript have drifted apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_REF_RE = re.compile(r"\\(?:page)?ref\{([^}]+)\}")
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
_BIBITEM_RE = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}")
_GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
_ENV_RE = re.compile(r"\\(begin|end)\{([^}]+)\}")


def strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


@dataclass
class CheckReport:
    undefined_refs: list[str] = field(default_factory=list)
    unused_labels: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    uncited_bibitems: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    unbalanced_environments: list[str] = field(default_factory=list)
    duplicate_labels: list[str] = field(default_factory=list)

    @property
    def problems(self) -> list[str]:
        """Findings that would break or visibly damage the compiled document."""
        out: list[str] = []
        out += [f"undefined reference: \\ref{{{r}}}" for r in self.undefined_refs]
        out += [f"missing citation key: {c}" for c in self.missing_citations]
        out += [f"missing file: {f}" for f in self.missing_files]
        out += [f"unbalanced environment: {e}" for e in self.unbalanced_environments]
        out += [f"duplicate label: {d}" for d in self.duplicate_labels]
        return out

    @property
    def warnings(self) -> list[str]:
        """Findings that compile cleanly but usually indicate an oversight."""
        out = [f"label defined but never referenced: {label}" for label in self.unused_labels]
        out += [f"bibliography entry never cited: {key}" for key in self.uncited_bibitems]
        return out


def _resolve(target: str, root: Path, paper_dir: Path, extensions: tuple[str, ...]) -> bool:
    """Whether a LaTeX path argument resolves, honouring the ``\\paperdir`` indirection."""
    # Both drivers compile the same body, so a path is satisfied if it resolves from either
    # the repository root or the paper directory.
    cleaned = target.replace("\\paperdir", "").strip()
    for base in (root, paper_dir):
        for suffix in ("", *extensions):
            if (base / f"{cleaned}{suffix}").exists():
                return True
    return False


def check_manuscript(
    sources: list[Path], root: Path, paper_dir: Path, bibliography: Path | None = None
) -> CheckReport:
    """Cross-check labels, citations and included files across ``sources``."""
    report = CheckReport()
    combined = ""
    for path in sources:
        if path.exists():
            combined += strip_comments(path.read_text(encoding="utf-8")) + "\n"

    labels: list[str] = _LABEL_RE.findall(combined)
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            report.duplicate_labels.append(label)
        seen.add(label)
    referenced = set(_REF_RE.findall(combined))
    report.undefined_refs = sorted(referenced - seen)
    report.unused_labels = sorted(seen - referenced)

    cited: set[str] = set()
    for group in _CITE_RE.findall(combined):
        cited.update(key.strip() for key in group.split(","))
    if bibliography and bibliography.exists():
        defined = set(_BIBITEM_RE.findall(strip_comments(
            bibliography.read_text(encoding="utf-8")
        )))
        report.missing_citations = sorted(cited - defined)
        report.uncited_bibitems = sorted(defined - cited)

    for target in _GRAPHIC_RE.findall(combined):
        if not _resolve(target, root, paper_dir, (".pdf", ".png", ".eps", ".jpg")):
            report.missing_files.append(target)
    for target in _INPUT_RE.findall(combined):
        if not _resolve(target, root, paper_dir, (".tex",)):
            report.missing_files.append(target)

    depth: dict[str, int] = {}
    for kind, name in _ENV_RE.findall(combined):
        depth[name] = depth.get(name, 0) + (1 if kind == "begin" else -1)
    report.unbalanced_environments = sorted(
        f"{name} ({count:+d})" for name, count in depth.items() if count != 0
    )
    return report
