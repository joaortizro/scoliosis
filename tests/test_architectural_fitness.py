"""Architectural fitness functions.

Cavekit: cavekit-model-exploration.md R8 / T-030.

Enforces:

- No `.cuda()` calls anywhere in the `ai/` package. All tensor / module
  placement MUST go through `.to(device)` so the code works on CUDA, ROCm,
  and CPU without edits.
- No literal `cuda` device construction — `torch.device("cuda")` is still
  allowed (PyTorch uses the "cuda" string on ROCm wheels too), but passing
  the literal index "cuda:0" via a hardcoded constant is discouraged in
  favour of the device helper.

The check walks the source tree, so it is cheap enough to run on every
pytest invocation.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = REPO_ROOT / "ai"
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _iter_py_files(root: Path):
    if not root.is_dir():
        return
    for p in root.rglob("*.py"):
        if any(part.startswith((".", "__pycache__")) for part in p.parts):
            continue
        yield p


def _find_cuda_calls(text: str) -> list[int]:
    """Return line numbers of real `.cuda(` tokens — skipping strings/comments.

    Uses Python's tokenizer so docstrings, string literals, and comments
    never trip the check.
    """
    hits: list[int] = []
    prev = None
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenizeError, IndentationError):
        return hits
    for tok in toks:
        if tok.type == tokenize.NAME and tok.string == "cuda":
            if prev is not None and prev.type == tokenize.OP and prev.string == ".":
                # Look ahead for "("
                idx = toks.index(tok)
                if idx + 1 < len(toks):
                    nxt = toks[idx + 1]
                    if nxt.type == tokenize.OP and nxt.string == "(":
                        hits.append(tok.start[0])
        prev = tok
    return hits


@pytest.mark.parametrize("root", [AI_ROOT, SCRIPTS_ROOT])
def test_no_dot_cuda_calls(root: Path) -> None:
    """No `.cuda()` calls in `ai/` or `scripts/` source files.

    Training, evaluation, and inference code paths must place tensors and
    modules via `.to(device)`. The device is chosen by
    `ai.utils.get_device()` in priority order CUDA → ROCm → CPU.
    """
    offenders: list[str] = []
    for py in _iter_py_files(root):
        try:
            text = py.read_text()
        except OSError:
            continue
        for lineno in _find_cuda_calls(text):
            offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}")
    assert not offenders, (
        "Found forbidden `.cuda()` calls — use `.to(device)` instead:\n  "
        + "\n  ".join(offenders)
    )
