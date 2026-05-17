"""Build the v2_corrected_x2 triage CSV — the authoritative list of cases needing
manual GIMP correction.

Source: data/processed/audit_v2_corrected/clean_index.csv (250 cases scanned
by notebooks/sandbox/data_exploration_v2.ipynb on the v2_corrected dataset).

Filter: keep cases where target_vertebrae_count < 17 AND status != 'excluded'.
These are the cases where Jorge's Phase-1 correction couldn't add back missing
vertebrae (Phase 1 only relabels fused IDs; it doesn't paint new pixels).

Output: data/processed/v2_corrected_x2_triage/audit_x2_triage.csv with one row
per case, ranked by missing-vertebra count (most-missing first) then by Cobb
severity. Each row identifies the source image + mask, which vertebra IDs are
missing, and provides the LabelMulti_* base_name used by mask_correction_workflow.py.

Usage:
    python scripts/mask_correction_x2/triage_audit_cases.py
    # Optional: write a human-readable per-person assignment too:
    python scripts/mask_correction_x2/triage_audit_cases.py --split-among 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO_ROOT / "data" / "processed" / "audit_v2_corrected"
OUT_DIR = REPO_ROOT / "data" / "processed" / "v2_corrected_x2_triage"
SOURCE_DATASET_DIR = REPO_ROOT / "data" / "raw" / "Scoliosis_Dataset_v2_corrected"

ID_TO_NAME = {
    1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5", 6: "T6",
    7: "T7", 8: "T8", 9: "T9", 10: "T10", 11: "T11", 12: "T12",
    13: "L1", 14: "L2", 15: "L3", 16: "L4", 17: "L5",
}
NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}
ALL_TARGET_IDS = set(ID_TO_NAME.keys())


def parse_present_ids(present_str: str) -> set[int]:
    """Parse 'target_vertebrae_present' — comma-separated vertebra names ('T1,T2,...')."""
    if pd.isna(present_str) or str(present_str).strip() == "":
        return set()
    out = set()
    for tok in str(present_str).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in NAME_TO_ID:
            raise ValueError(f"Unknown vertebra name in target_vertebrae_present: {tok!r}")
        out.add(NAME_TO_ID[tok])
    return out


def base_name_for(row: pd.Series) -> str:
    """LabelMulti_S_156 from the multiclass_mask_path."""
    return Path(row["multiclass_mask_path"]).stem


def gap_pattern(missing_ids: set[int]) -> str:
    """Classify the gap shape — informs the GIMP fix difficulty.

    - 'l5_tail'      : only L5 missing → fix last vertebra; trivial if visible
    - 'lumbar_tail'  : missing only L4 and/or L5 → tail-end gap
    - 'thoracic_top' : only T1/T2 missing → top-of-spine gap
    - 'mid_only'     : missing block entirely in the middle (T6-L1) → mid-spine
    - 'mixed'        : multiple gap regions or scattered
    - 'empty'        : nothing missing (shouldn't appear in triage output)
    """
    if not missing_ids:
        return "empty"
    if missing_ids == {17}:
        return "l5_tail"
    if missing_ids.issubset({16, 17}):
        return "lumbar_tail"
    if missing_ids.issubset({1, 2}):
        return "thoracic_top"
    if missing_ids.issubset(set(range(6, 14))):  # T6..L1
        return "mid_only"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-index",
        default=str(AUDIT_DIR / "clean_index.csv"),
        help="Path to the audit's clean_index.csv (default: data/processed/audit_v2_corrected/clean_index.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory for the triage CSVs (default: data/processed/v2_corrected_x2_triage/)",
    )
    parser.add_argument(
        "--split-among",
        type=int,
        default=0,
        help="If > 0, also write per-person assignment files (round-robin).",
    )
    parser.add_argument(
        "--people",
        nargs="+",
        default=["Jorge", "Fedys", "Beto", "Jonas"],
        help="Names for --split-among (default: Jorge Fedys Beto Jonas).",
    )
    args = parser.parse_args()

    clean_index = pd.read_csv(args.clean_index)
    print(f"Loaded {len(clean_index)} cases from {args.clean_index}")

    # Filter: need correction iff target_vertebrae_count < 17 AND not 'excluded'.
    # (excluded cases have target_vertebrae_count == 0 — fully unusable.)
    needs_fix = clean_index[
        (clean_index["target_vertebrae_count"] < 17)
        & (clean_index["status"] != "excluded")
    ].copy()
    print(f"Cases needing manual correction (count < 17, not excluded): {len(needs_fix)}")

    # Compute per-case gap info.
    needs_fix["present_ids"] = needs_fix["target_vertebrae_present"].apply(parse_present_ids)
    needs_fix["missing_ids"] = needs_fix["present_ids"].apply(lambda s: ALL_TARGET_IDS - s)
    needs_fix["missing_count"] = needs_fix["missing_ids"].apply(len)
    needs_fix["missing_vertebrae_ids"] = needs_fix["missing_ids"].apply(
        lambda s: ";".join(str(i) for i in sorted(s))
    )
    needs_fix["missing_vertebrae_names"] = needs_fix["missing_ids"].apply(
        lambda s: ";".join(ID_TO_NAME[i] for i in sorted(s))
    )
    needs_fix["gap_pattern"] = needs_fix["missing_ids"].apply(gap_pattern)
    needs_fix["base_name"] = needs_fix.apply(base_name_for, axis=1)

    # Verify all paths point INSIDE the source dataset dir (defense-in-depth).
    source_str = str(SOURCE_DATASET_DIR)
    not_in_source = needs_fix[
        ~needs_fix["multiclass_mask_path"].astype(str).str.startswith(source_str)
    ]
    if len(not_in_source):
        raise SystemExit(
            f"Refusing to triage: {len(not_in_source)} mask paths fall outside the "
            f"source dataset dir {source_str}. Audit data inconsistent."
        )

    # Sort by missing_count DESC, then Cobb angle DESC (severity tiebreaker).
    needs_fix = needs_fix.sort_values(
        by=["missing_count", "cobb_angle_deg"],
        ascending=[False, False],
    ).reset_index(drop=True)
    needs_fix["triage_order"] = needs_fix.index + 1

    # Compact output columns.
    out_cols = [
        "triage_order",
        "patient_id",
        "category",
        "status",
        "cobb_angle_deg",
        "target_vertebrae_count",
        "missing_count",
        "missing_vertebrae_ids",
        "missing_vertebrae_names",
        "gap_pattern",
        "base_name",
        "image_path",
        "multiclass_mask_path",
        "issues",
    ]
    out = needs_fix[out_cols]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "audit_x2_triage.csv"
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} triage rows → {out_csv}")

    # Per-pattern + per-count rollups for quick reading.
    summary = {
        "total_cases": int(len(out)),
        "by_missing_count": out["missing_count"].value_counts().sort_index().to_dict(),
        "by_gap_pattern": out["gap_pattern"].value_counts().to_dict(),
        "by_category": out["category"].value_counts().to_dict(),
        "estimated_gimp_minutes_per_case": {"easy": 5, "medium": 10, "heavy": 20},
        "estimated_total_hours_low": round(len(out) * 5 / 60, 1),
        "estimated_total_hours_high": round(len(out) * 20 / 60, 1),
    }
    summary_path = out_dir / "audit_x2_triage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=int))
    print(f"Wrote summary → {summary_path}")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Optional: per-person split.
    if args.split_among > 0:
        people = args.people[: args.split_among]
        if len(people) != args.split_among:
            raise SystemExit(
                f"--split-among={args.split_among} but only {len(people)} names "
                f"provided via --people. Pass enough names."
            )
        # Round-robin assignment by triage_order — balances heavy cases across people.
        assigned = out.assign(
            assigned_to=[people[i % args.split_among] for i in range(len(out))]
        )
        for person in people:
            slice_df = assigned[assigned["assigned_to"] == person]
            person_csv = out_dir / f"audit_x2_assignment_{person.lower()}.csv"
            slice_df.to_csv(person_csv, index=False)
            print(f"  {person}: {len(slice_df)} cases → {person_csv}")

    print("\n=== Top 10 ===")
    print(out.head(10)[["triage_order", "base_name", "missing_count",
                        "missing_vertebrae_names", "gap_pattern",
                        "cobb_angle_deg"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
