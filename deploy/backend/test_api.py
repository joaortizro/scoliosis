"""
test_api.py — Prueba rápida de la API de segmentación
Uso:
    python test_api.py imagen.jpg
    python test_api.py imagen.jpg --conf 0.05 --endpoint t1t8-only
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("── Health ──────────────────────────────")
    print(json.dumps(r.json(), indent=2))
    print()


def segment(image_path: str, conf: float = 0.01, endpoint: str = "segment"):
    path = Path(image_path)
    if not path.exists():
        print(f"❌ Archivo no encontrado: {image_path}")
        sys.exit(1)

    url = f"{BASE_URL}/{endpoint}?conf={conf}&return_image=true"
    with open(path, "rb") as f:
        r = requests.post(url, files={"file": (path.name, f, "image/jpeg")})

    if r.status_code != 200:
        print(f"❌ Error {r.status_code}: {r.text}")
        sys.exit(1)

    data = r.json()
    print(f"── Resultado ({endpoint}) ──────────────────")
    print(f"  Modo:          {data['model_used']}")
    print(f"  Dispositivo:   {data['device']}")
    print(f"  Detectadas:    {data['total_detected']} vértebras")
    print()
    print(f"  {'Vértebra':<10} {'Confianza':>10} {'Cx':>8} {'Cy':>8} {'Área':>8} {'Fuente'}")
    print(f"  {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*12}")
    for v in data["vertebrae"]:
        print(f"  {v['label']:<10} {v['confidence']:>10.4f} {v['centroid_x']:>8.1f} "
              f"{v['centroid_y']:>8.1f} {v['area_px']:>8} {v['source']}")

    # Guardar imagen segmentada
    if data.get("image_base64"):
        out_path = path.stem + f"_segmented_{endpoint}.png"
        img_bytes = base64.b64decode(data["image_base64"])
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"\n  ✅ Imagen guardada en: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Ruta a la radiografía")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument(
        "--endpoint",
        choices=["segment", "segment/full-only", "segment/t1t8-only"],
        default="segment",
    )
    args = parser.parse_args()

    test_health()
    segment(args.image, args.conf, args.endpoint)
