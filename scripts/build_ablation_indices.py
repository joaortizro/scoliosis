"""Build the 3 clean_index variants for the dataset ablation experiment.

Produces:
  data/processed/audit_v2_corrected/clean_index.csv                          (existing, untouched — baseline D0)
  data/processed/audit_v2_corrected_x2/clean_index.csv                       (D1 — v2 with 6 mask overrides)
  data/processed/audit_v2_corrected_x2_plus_roboflow/clean_index.csv         (D2 — D1 + 18 extra_roboflow cases)

The trainer's `run(cfg)` will be called with `cfg["data"]["clean_index"]`
pointed at each of these per ablation variant.

Schema is the same as the existing v2 clean_index (status, target_vertebrae_count,
target_vertebrae_present, image_path, multiclass_mask_path, …) so all the
existing split + trainer code Just Works.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_INDEX = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "clean_index.csv"
X2_MASK_DIR = REPO_ROOT / "data" / "raw" / "Scoliosis_Dataset_v2_corrected_x2" / "LabelMultiClass_ID_PNG"
ROBOFLOW_INDEX = REPO_ROOT / "data" / "raw" / "Scoliosis_Dataset_extra_roboflow" / "indice_dataset.csv"

ID_TO_NAME = {1:"T1",2:"T2",3:"T3",4:"T4",5:"T5",6:"T6",7:"T7",8:"T8",9:"T9",10:"T10",11:"T11",12:"T12",13:"L1",14:"L2",15:"L3",16:"L4",17:"L5"}


def to_repo_relative(path: str | Path) -> str:
    """Strip any /home/<user>/scoliosis/ prefix, return repo-relative POSIX path.

    The trainer's CWD is the repo root, so relative paths Just Work across machines
    (local /home/ortiz/scoliosis vs EC2 /home/ec2-user/scoliosis).
    """
    s = str(path)
    # Anchor on the 'data/' or 'ai/' or 'scripts/' segment — repo-internal roots
    for marker in ("/data/", "/ai/", "/scripts/", "/experiments/"):
        if marker in s:
            return s[s.index(marker) + 1 :]  # drop the leading slash on marker
    return s  # already relative or some non-repo path


def mask_completeness(mask_path: Path) -> tuple[int, str]:
    """Return (count, comma-separated names) from a mask file."""
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    present = sorted(set(np.unique(arr).tolist()) - {0})
    return len(present), ",".join(ID_TO_NAME[i] for i in present)


def build_x2() -> pd.DataFrame:
    """v2_corrected with 6 mask overrides applied (target counts refreshed to 17)."""
    base = pd.read_csv(BASELINE_INDEX)
    out = base.copy()

    # First, rewrite all path columns to repo-relative so the file is
    # portable across machines (local /home/ortiz vs EC2 /home/ec2-user).
    for col in ("image_path", "binary_mask_path", "multiclass_mask_path",
                "curve_csv_path", "overlay_path"):
        if col in out.columns:
            out[col] = out[col].astype(str).apply(to_repo_relative)

    overridden = []
    for mask_file in sorted(X2_MASK_DIR.glob("LabelMulti_*.png")):
        base_name = mask_file.stem  # e.g. LabelMulti_N_23
        prefix, kind, num = base_name.split("_")  # LabelMulti, N, 23
        patient_id = int(num)
        category = "Normal" if kind == "N" else "Scoliosis"

        row_mask = (out["patient_id"] == patient_id) & (out["category"] == category)
        n_match = int(row_mask.sum())
        if n_match != 1:
            raise RuntimeError(f"Expected exactly 1 row for {base_name}, found {n_match}")

        new_count, new_present = mask_completeness(mask_file)
        out.loc[row_mask, "multiclass_mask_path"] = to_repo_relative(str(mask_file.resolve()))
        out.loc[row_mask, "target_vertebrae_count"] = new_count
        out.loc[row_mask, "target_vertebrae_present"] = new_present
        if new_count == 17:
            issues_cell = out.loc[row_mask, "issues"].iloc[0]
            if pd.isna(issues_cell) or str(issues_cell).strip() == "":
                out.loc[row_mask, "status"] = "ok"
        overridden.append((base_name, new_count))

    print(f"Overrode {len(overridden)} cases with x2 corrected masks:")
    for n, c in overridden:
        print(f"  {n} → target_vertebrae_count={c}")
    return out


def build_x2_plus_roboflow(x2_df: pd.DataFrame) -> pd.DataFrame:
    """x2 with 18 extra_roboflow cases appended (Spanish↔English schema reconciliation)."""
    rb = pd.read_csv(ROBOFLOW_INDEX)

    # Map roboflow's English schema to v2's clean_index schema.
    # v2 columns: patient_id, category, image_path, binary_mask_path,
    #             multiclass_mask_path, curve_csv_path, overlay_path,
    #             image_h, image_w, cobb_angle_deg, target_vertebrae_count,
    #             target_vertebrae_present, status, issues
    rb_mapped = pd.DataFrame({
        "patient_id": rb["patient_id"],
        "category": rb["category"],
        "image_path": rb["image_path"].astype(str).apply(to_repo_relative),
        "binary_mask_path": "",                      # not provided by roboflow
        "multiclass_mask_path": rb["multiclass_mask_path"].astype(str).apply(to_repo_relative),
        "curve_csv_path": "",                        # not provided
        "overlay_path": "",                          # not provided
        "image_h": rb["image_h"],
        "image_w": rb["image_w"],
        "cobb_angle_deg": np.nan,                    # roboflow has no Cobb GT
        "target_vertebrae_count": rb["target_vertebrae_count"],
        "target_vertebrae_present": rb["target_vertebrae_present"],
        "status": rb["status"],
        "issues": rb["issues"].fillna(""),
    })

    # Sanity: no patient_id collision between v2 and roboflow.
    v2_n_ids = set(x2_df[x2_df["category"] == "Normal"]["patient_id"])
    rb_n_ids = set(rb_mapped[rb_mapped["category"] == "Normal"]["patient_id"])
    n_collisions = v2_n_ids & rb_n_ids
    v2_s_ids = set(x2_df[x2_df["category"] == "Scoliosis"]["patient_id"])
    rb_s_ids = set(rb_mapped[rb_mapped["category"] == "Scoliosis"]["patient_id"])
    s_collisions = v2_s_ids & rb_s_ids
    if n_collisions or s_collisions:
        raise RuntimeError(
            f"Patient ID collision: Normal={n_collisions}, Scoliosis={s_collisions}"
        )

    merged = pd.concat([x2_df, rb_mapped], ignore_index=True)
    print(f"Appended {len(rb_mapped)} extra_roboflow cases → total {len(merged)} cases")
    return merged


def main() -> int:
    print("=== Building D1: v2_corrected_x2 (250 cases, 6 mask overrides) ===")
    x2_df = build_x2()
    x2_out = REPO_ROOT / "data" / "processed" / "audit_v2_corrected_x2"
    x2_out.mkdir(parents=True, exist_ok=True)
    x2_csv = x2_out / "clean_index.csv"
    x2_df.to_csv(x2_csv, index=False)
    print(f"Wrote {x2_csv}")
    print(f"  status counts: {x2_df['status'].value_counts().to_dict()}")
    print()

    print("=== Building D2: v2_corrected_x2 + extra_roboflow (268 cases) ===")
    full_df = build_x2_plus_roboflow(x2_df)
    full_out = REPO_ROOT / "data" / "processed" / "audit_v2_corrected_x2_plus_roboflow"
    full_out.mkdir(parents=True, exist_ok=True)
    full_csv = full_out / "clean_index.csv"
    full_df.to_csv(full_csv, index=False)
    print(f"Wrote {full_csv}")
    print(f"  status counts: {full_df['status'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
