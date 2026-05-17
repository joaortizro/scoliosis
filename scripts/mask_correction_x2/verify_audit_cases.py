"""Generate a single self-contained HTML page to verify the 69-case triage.

For each candidate case from audit_x2_triage.csv:
  - radiograph (grayscale)
  - current multiclass mask (bright RGB via palette)
  - radiograph + mask overlay
  - metadata: missing vertebrae, gap pattern, Cobb angle
  - 3 radio choices: needs_fix / already_complete / uncorrectable
  - notes textbox

The page bundles all PNGs as base64 (no external file deps). A "Download CSV"
button collects your choices + notes into verification_results.csv.

Output: .local/mask_correction_x2/verify/verify_all.html

Usage:
    python scripts/mask_correction_x2/verify_audit_cases.py
    # Open the file in a browser (file:// is fine — everything is inline).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIAGE = REPO_ROOT / "data" / "processed" / "v2_corrected_x2_triage" / "audit_x2_triage.csv"
DEFAULT_OUT = REPO_ROOT / ".local" / "mask_correction_x2" / "verify" / "verify_all.html"

ID_TO_HEX: Dict[int, str] = {
    0: "#000000",
    1: "#F2D10C",  2: "#EBF20C",  3: "#C4F20C",  4: "#9CF20C",
    5: "#75F20C",  6: "#4DF20C",  7: "#26F20C",  8: "#0CF219",
    9: "#0CF240",  10: "#0CF268", 11: "#0CF28F", 12: "#0CF2B7",
    13: "#0CF2DE", 14: "#0CDEF2", 15: "#0CB7F2", 16: "#0C8FF2",
    17: "#0C68F2",
}
ID_TO_NAME = {
    0: "bg",
    1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5", 6: "T6",
    7: "T7", 8: "T8", 9: "T9", 10: "T10", 11: "T11", 12: "T12",
    13: "L1", 14: "L2", 15: "L3", 16: "L4", 17: "L5",
}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


ID_TO_RGB = {k: hex_to_rgb(v) for k, v in ID_TO_HEX.items()}


def mask_to_color(mask_arr: np.ndarray) -> Image.Image:
    h, w = mask_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in ID_TO_RGB.items():
        rgb[mask_arr == class_id] = color
    return Image.fromarray(rgb, mode="RGB")


def make_overlay(rad_img: Image.Image, mask_img: Image.Image, alpha: float = 0.55) -> Image.Image:
    rad = rad_img.convert("RGB")
    mask = mask_img.convert("RGB")
    if mask.size != rad.size:
        mask = mask.resize(rad.size, Image.NEAREST)
    rad_arr = np.array(rad, dtype=np.float32)
    mask_arr = np.array(mask, dtype=np.float32)
    # Only blend where mask has non-zero pixels.
    nonzero = (mask_arr.sum(axis=-1) > 0)[..., None]
    blended = np.where(nonzero, (1 - alpha) * rad_arr + alpha * mask_arr, rad_arr)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def img_to_base64_png(img: Image.Image, max_w: int = 320) -> str:
    if img.width > max_w:
        ratio = max_w / img.width
        new_size = (max_w, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_radiograph(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode != "L" and img.mode != "RGB":
        img = img.convert("L")
    return img.copy()


def load_mask(mask_path: str) -> np.ndarray:
    img = Image.open(mask_path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def render_case_html(row: pd.Series, max_w: int = 320) -> str:
    rad_img = load_radiograph(row["image_path"])
    mask_arr = load_mask(row["multiclass_mask_path"])
    color_mask = mask_to_color(mask_arr)
    overlay = make_overlay(rad_img, color_mask)

    rad_b64 = img_to_base64_png(rad_img, max_w=max_w)
    mask_b64 = img_to_base64_png(color_mask, max_w=max_w)
    overlay_b64 = img_to_base64_png(overlay, max_w=max_w)

    present_ids = sorted(set(int(x) for x in np.unique(mask_arr) if x != 0))
    present_names = ", ".join(ID_TO_NAME[i] for i in present_ids)

    missing = html.escape(str(row["missing_vertebrae_names"]))
    pattern = html.escape(str(row["gap_pattern"]))
    base = html.escape(str(row["base_name"]))
    cat = html.escape(str(row["category"]))
    cobb = (
        f"{row['cobb_angle_deg']:.1f}°"
        if not pd.isna(row.get("cobb_angle_deg"))
        else "N/A"
    )
    issues = html.escape(str(row.get("issues") or "") or "—")
    order = int(row["triage_order"])

    return f"""
<div class="case" data-base="{base}" data-order="{order}">
  <div class="case-header">
    <span class="order">#{order}</span>
    <span class="base">{base}</span>
    <span class="meta">{cat} · Cobb {cobb} · pattern <b>{pattern}</b></span>
  </div>
  <div class="case-meta">
    <div><b>Missing:</b> <span class="missing">{missing}</span></div>
    <div><b>Present ({len(present_ids)}/17):</b> {html.escape(present_names)}</div>
    <div><b>Audit issues:</b> {issues}</div>
  </div>
  <div class="images">
    <figure><img src="data:image/png;base64,{rad_b64}" alt="radiograph"/><figcaption>radiograph</figcaption></figure>
    <figure><img src="data:image/png;base64,{mask_b64}" alt="mask"/><figcaption>current mask (color)</figcaption></figure>
    <figure><img src="data:image/png;base64,{overlay_b64}" alt="overlay"/><figcaption>overlay</figcaption></figure>
  </div>
  <div class="decision">
    <label><input type="radio" name="decision_{order}" value="needs_fix"> needs_fix</label>
    <label><input type="radio" name="decision_{order}" value="already_complete"> already_complete</label>
    <label><input type="radio" name="decision_{order}" value="anatomical_variant"> anatomical_variant</label>
    <label><input type="radio" name="decision_{order}" value="uncorrectable"> uncorrectable</label>
    <label><input type="radio" name="decision_{order}" value=""> (skip)</label>
    <input type="text" class="notes" name="notes_{order}" placeholder="notes (e.g. 'sacralization', 'cropped FOV')"/>
  </div>
</div>
"""


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2_corrected_x2 verification — {n} cases</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 0 24px 80px 24px; }}
  header {{ position: sticky; top: 0; background: #111; padding: 16px 0; border-bottom: 1px solid #333; z-index: 10; }}
  header h1 {{ margin: 0 0 6px 0; font-size: 18px; }}
  header .stats {{ font-size: 13px; color: #aaa; }}
  header button {{ background: #2a7; color: white; border: 0; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; margin-left: 8px; }}
  header button:hover {{ background: #3b8; }}
  header .toolbar {{ margin-top: 8px; }}
  .case {{ border: 1px solid #333; border-radius: 6px; padding: 12px 16px; margin: 18px 0; background: #1a1a1a; }}
  .case-header {{ display: flex; gap: 12px; align-items: baseline; margin-bottom: 6px; }}
  .case-header .order {{ color: #888; font-family: monospace; }}
  .case-header .base {{ font-weight: bold; font-family: monospace; color: #ffd; }}
  .case-header .meta {{ color: #aaa; font-size: 13px; }}
  .case-meta {{ font-size: 13px; color: #ccc; margin-bottom: 8px; }}
  .case-meta .missing {{ color: #f87; font-weight: bold; }}
  .images {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  figure {{ margin: 0; text-align: center; }}
  figure img {{ display: block; max-width: 320px; border: 1px solid #444; }}
  figure figcaption {{ font-size: 11px; color: #888; margin-top: 4px; }}
  .decision {{ margin-top: 10px; display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
  .decision label {{ font-size: 13px; cursor: pointer; }}
  .decision input[type="text"] {{ flex: 1; min-width: 200px; padding: 4px 8px; background: #222; color: #eee; border: 1px solid #444; border-radius: 3px; }}
</style>
</head>
<body>
<header>
  <h1>v2_corrected_x2 — verify {n} candidate cases</h1>
  <div class="stats">Per-case: pick <code>needs_fix</code>, <code>already_complete</code>, <code>anatomical_variant</code> (sacralization / agenesis / transitional vertebra — patient genuinely has &lt;17), or <code>uncorrectable</code> (image FOV cropped / occluded). Skipped rows export with empty decision.</div>
  <div class="toolbar">
    <span id="progress">0 / {n} decided</span>
    <button onclick="downloadCsv()">Download verification_results.csv</button>
  </div>
</header>
<main>
{cases}
</main>
<script>
const N = {n};

function updateProgress() {{
  const decided = document.querySelectorAll('input[type=radio]:checked').length;
  document.getElementById('progress').textContent = decided + ' / ' + N + ' decided';
}}
document.addEventListener('change', updateProgress);

function downloadCsv() {{
  const rows = [['triage_order', 'base_name', 'decision', 'notes']];
  document.querySelectorAll('.case').forEach(div => {{
    const order = div.dataset.order;
    const base = div.dataset.base;
    const checked = div.querySelector('input[type=radio]:checked');
    const decision = checked ? checked.value : '';
    const notes = div.querySelector('input.notes').value || '';
    rows.push([order, base, decision, notes]);
  }});
  const csv = rows.map(r => r.map(c => '"' + String(c).replace(/"/g, '""') + '"').join(',')).join('\\n');
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'verification_results.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triage", default=str(DEFAULT_TRIAGE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-w", type=int, default=320, help="Max image width in HTML.")
    parser.add_argument("--limit", type=int, default=0, help="Render only first N cases (smoke test).")
    args = parser.parse_args()

    triage = pd.read_csv(args.triage)
    if args.limit > 0:
        triage = triage.head(args.limit)
    print(f"Rendering {len(triage)} cases from {args.triage}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    case_blocks = []
    for i, row in triage.iterrows():
        try:
            case_blocks.append(render_case_html(row, max_w=args.max_w))
        except Exception as e:
            print(f"  [FAILED] order={row['triage_order']} {row['base_name']}: {e}")
        if (i + 1) % 10 == 0:
            print(f"  rendered {i + 1}/{len(triage)}")

    page = PAGE_TEMPLATE.format(n=len(case_blocks), cases="\n".join(case_blocks))
    out_path.write_text(page, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")
    print(f"Open in your browser: file://{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
