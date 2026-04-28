"""Architectural fitness check for ``ai/visualization/``.

Cavekit: cavekit-case-visualization.md R6 / T-110.

The visualization package must remain a pure rendering library — importable
from any context (notebook, CLI, future inference adapter, DVC stage)
without dragging in DVC, MLflow, FastAPI, or SQLAlchemy. Those frameworks
belong in their own adapters; pulling them into the rendering module
would couple visualization to deployment infrastructure and break the
"pure library" contract documented in the module docstring.

The check parses every ``.py`` file under ``ai/visualization/`` with the
``ast`` module so docstrings, comments, and string literals that mention
those framework names cannot trip a false positive — only real ``import``
or ``from ... import`` statements count.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VISUALIZATION_ROOT = REPO_ROOT / "ai" / "visualization"

#: Top-level package names that ``ai/visualization/`` must never import.
FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {"dvc", "mlflow", "fastapi", "sqlalchemy"}
)


def _iter_py_files(root: Path):
    if not root.is_dir():
        return
    for p in root.rglob("*.py"):
        if any(part.startswith((".", "__pycache__")) for part in p.parts):
            continue
        yield p


def _imported_roots(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, top-level module) for every real import in the AST.

    For ``import foo.bar`` and ``import foo.bar as baz`` this yields
    ``(lineno, "foo")``. For ``from foo.bar import baz`` it yields
    ``(lineno, "foo")``. Relative imports (``from . import x``) are
    skipped — they cannot reference top-level forbidden packages.
    """

    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    out.append((node.lineno, alias.name.split(".", 1)[0]))
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports — module is None or level > 0.
            if node.level and node.level > 0:
                continue
            if node.module:
                out.append((node.lineno, node.module.split(".", 1)[0]))
    return out


def test_visualization_has_no_forbidden_framework_imports() -> None:
    """``ai/visualization/`` must not import dvc/mlflow/fastapi/sqlalchemy.

    The cavekit (R6) requires the rendering library to stay framework-free
    so the same module can be re-used by notebooks, the CLI, the eventual
    FastAPI inference adapter, and DVC stages without circular coupling.
    """

    offenders: list[str] = []
    for py in _iter_py_files(VISUALIZATION_ROOT):
        try:
            source = py.read_text()
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=str(py))
        except SyntaxError as exc:  # pragma: no cover — would already break pytest collection
            offenders.append(f"{py.relative_to(REPO_ROOT)}: parse error: {exc}")
            continue
        for lineno, root_module in _imported_roots(tree):
            if root_module in FORBIDDEN_IMPORT_ROOTS:
                offenders.append(
                    f"{py.relative_to(REPO_ROOT)}:{lineno} imports forbidden "
                    f"module '{root_module}'"
                )
    assert not offenders, (
        "ai/visualization/ must not import dvc, mlflow, fastapi, or "
        "sqlalchemy — these frameworks belong in adapters, not in the "
        "pure rendering library:\n  " + "\n  ".join(offenders)
    )
