"""Run-directory bookkeeping (cfg-hash cache, reproducibility files).

Every training run is keyed by a deterministic hash over its config so
that re-running with the same params reuses the cached checkpoint
instead of retraining. Run dirs carry enough metadata
(``git_sha.txt``, ``pip_freeze.txt``, ``cuda.txt``, ``python.txt``,
``RUN.md``) for a future agent to reproduce the number months later.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def find_cached_run(cfg: dict, root: Path) -> Path | None:
    """Return the most recent run dir whose hash matches ``cfg``, else None."""
    if not root.exists():
        return None
    h = config_hash(cfg)
    matches = sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith(h)])
    return matches[-1] if matches else None


def find_inflight_run(cfg: dict, root: Path) -> Path | None:
    """Return the most recent run dir for ``cfg`` that has a ``last.pt``
    epoch checkpoint but no final ``metrics.json`` — i.e. an interrupted
    run that can be resumed. Returns None if no such dir exists."""
    if not root.exists():
        return None
    h = config_hash(cfg)
    matches = sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith(h)])
    for p in reversed(matches):
        if (p / "last.pt").exists() and not (p / "metrics.json").exists():
            return p
    return None


def new_run_dir(cfg: dict, root: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = root / f"{stamp}_{config_hash(cfg)}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _pip_freeze() -> str:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        return ""


def _cuda_info() -> str:
    if torch.cuda.is_available():
        return f"cuda_available=True\ndevice_count={torch.cuda.device_count()}\nname={torch.cuda.get_device_name(0)}"
    return "cuda_available=False"


def write_provenance(run_dir: Path) -> None:
    (run_dir / "git_sha.txt").write_text(_git_sha())
    (run_dir / "pip_freeze.txt").write_text(_pip_freeze())
    (run_dir / "cuda.txt").write_text(_cuda_info())
    (run_dir / "python.txt").write_text(f"{platform.python_version()}\n{sys.executable}\n")


def save_run(
    run_dir: Path,
    state_dict: dict,
    history_rows: list[dict],
    cfg: dict,
    metrics: dict,
) -> None:
    import pandas as pd

    torch.save(state_dict, run_dir / "model.pt")
    pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)
    (run_dir / "cfg.json").write_text(json.dumps(cfg, indent=2, default=str))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_provenance(run_dir)
    write_run_md(run_dir, cfg, metrics)


def load_run(run_dir: Path) -> tuple[dict, list[dict], dict, dict]:
    import pandas as pd

    state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    history = pd.read_csv(run_dir / "history.csv").to_dict(orient="records")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = json.loads((run_dir / "cfg.json").read_text())
    return state, history, metrics, cfg


def write_run_md(run_dir: Path, cfg: dict, metrics: dict) -> None:
    """Human-readable summary for future agents browsing the run dir."""
    lines = [
        f"# Run {run_dir.name}",
        "",
        "## Config",
        "```json",
        json.dumps(cfg, indent=2, default=str),
        "```",
        "",
        "## Metrics",
        "```json",
        json.dumps(metrics, indent=2, default=str),
        "```",
        "",
        "## Reproduce",
        "1. `git checkout $(cat git_sha.txt)`",
        "2. `pip install -r pip_freeze.txt` (or just `requirements.txt`)",
        "3. `python scripts/train.py`",
        "",
        "Artifacts: `model.pt`, `history.csv`, `cfg.json`, `metrics.json`.",
    ]
    (run_dir / "RUN.md").write_text("\n".join(lines))
