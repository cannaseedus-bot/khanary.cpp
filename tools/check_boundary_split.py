"""Conformance check for source/contracts vs generated-artifacts boundaries."""
from __future__ import annotations

import re
from pathlib import Path
from conformance_common import verdict


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = ("build/", "_build/", ".build-copilot/", "out/")


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    cases: list[tuple[str, bool]] = []
    try:
        gitignore = ROOT / ".gitignore"
        rgignore = ROOT / ".rgignore"
        git_lines = load_lines(gitignore)
        rg_lines = load_lines(rgignore)

        cases.append(("gitignore_present", gitignore.exists()))
        cases.append(("rgignore_present", rgignore.exists()))

        for entry in GENERATED_DIRS:
            cases.append((f"gitignore_has_{entry.rstrip('/')}", entry in git_lines))
            cases.append((f"rgignore_has_{entry.rstrip('/')}", entry in rg_lines))

        include_generated = re.compile(
            r'#\s*include\s*[<"](?:\.\.?[\\/])?(?:build|_build|\.build-copilot|out)[\\/].*[>"]'
        )
        violations: list[str] = []
        for path in ROOT.joinpath("src").rglob("*"):
            if path.suffix.lower() not in {".h", ".hpp", ".cpp", ".cc", ".cxx"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if include_generated.search(text):
                violations.append(path.relative_to(ROOT).as_posix())

        cases.append(("source_includes_generated_artifacts", len(violations) == 0))
        return verdict("boundary_split", cases)
    except Exception as exc:
        return verdict("boundary_split", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
