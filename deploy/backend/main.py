"""
Spine Segmentation API — FastAPI
Modelos: YOLOv8s-seg completo (T1–L5) + especializado (T1–T8)
Ensemble: modelo T1-T8 para región superior (42%), completo para región inferior
"""

import io
import os
import base64
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
from ultralytics import YOLO

# ─── Configuración ────────────────────────────────────────────────────────────

IMG_SIZE   = 512
CROP_RATIO = 0.42          # Fracción superior usada por el modelo T1-T8
CROP_H     = int(IMG_SIZE * CROP_RATIO)  # 215 px

VERTEBRA_LABELS = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
# Colores BGR por región
COLORS = {
    "T":  (100, 200, 255),   # azul claro  → torácicas
    "L":  (100, 255, 150),   # verde claro → lumbares
}

MODEL_FULL_PATH = Path("spine_yolo/yolov8s_seg_768/weights/best.pt")
MODEL_T1T8_PATH = Path("spine_yolo/yolov8s_seg_t1t8/weights/best.pt")

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Spine Segmentation API",
    description=(
        "Segmentación de vértebras en radiografías de columna vertebral. "
        "Usa un ensemble YOLOv8s-seg: modelo especializado T1-T8 para la región "
        "torácica superior y modelo completo para el resto (T8-L5)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Modelos (cargados al arrancar) ───────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
model_full: Optional[YOLO] = None
model_t1t8: Optional[YOLO] = None

# RB-UNet ensemble (D2 5-fold). Loaded lazily on startup if the project's
# checkpoint sentinel is available. The deploy keeps working without it.
RBUNET_REPO_ROOT = Path(os.environ.get("SCOLIOSIS_REPO_ROOT", Path(__file__).resolve().parents[2]))
rbunet_ensemble = None  # type: ignore[var-annotated]


@app.on_event("startup")
async def load_models():
    global model_full, model_t1t8, rbunet_ensemble
    if MODEL_FULL_PATH.exists():
        model_full = YOLO(str(MODEL_FULL_PATH))
        print(f"✅ Modelo completo cargado desde {MODEL_FULL_PATH}")
    else:
        print(f"⚠️  Modelo completo NO encontrado en {MODEL_FULL_PATH}")

    if MODEL_T1T8_PATH.exists():
        model_t1t8 = YOLO(str(MODEL_T1T8_PATH))
        print(f"✅ Modelo T1-T8 cargado desde {MODEL_T1T8_PATH}")
    else:
        print(f"⚠️  Modelo T1-T8 NO encontrado en {MODEL_T1T8_PATH}")

    try:
        from rbunet import RBUNetEnsemble, resolve_d2_5fold
        cfg = resolve_d2_5fold(RBUNET_REPO_ROOT)
        rbunet_ensemble = RBUNetEnsemble(cfg)
        print(f"✅ RB-UNet D2 5-fold cargado ({rbunet_ensemble.n_folds} folds) desde {RBUNET_REPO_ROOT}")
    except Exception as e:
        print(f"⚠️  RB-UNet NO cargado: {e}")


# ─── Esquemas de respuesta ────────────────────────────────────────────────────

class VertebraResult(BaseModel):
    label: str          # "T1", "L3", …
    confidence: float   # 0-1
    centroid_x: float   # px en imagen redimensionada (512×512)
    centroid_y: float
    area_px: int        # área de la máscara en px²
    source: str         # "t1t8_model" | "full_model"


class SegmentationResponse(BaseModel):
    vertebrae: list[VertebraResult]
    total_detected: int
    image_base64: str   # PNG segmentado codificado en base64
    model_used: str     # "ensemble" | "full_only" | "t1t8_only"
    device: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_image(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (img_display, img_infer), ambas BGR uint8 a IMG_SIZE×IMG_SIZE.

    - img_display : original sin procesar → base de la imagen segmentada final
    - img_infer   : normalizada + CLAHE   → entra al modelo YOLO

    Separar ambas versiones evita que el preprocesamiento distorsione la
    imagen que ve el usuario.
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")

    # ── Profundidad de bits → uint8 ──────────────────────────────────────────
    if img.dtype == np.uint16:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # ── BGR 3 canales ────────────────────────────────────────────────────────
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ── img_display: solo redimensionar, sin más procesado ───────────────────
    img_display = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # ── img_infer: normalizar rango + CLAHE (solo si necesario) ─────────────
    gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
    p1   = float(np.percentile(gray, 1))
    p99  = float(np.percentile(gray, 99))

    img_infer = img_display.copy()
    if p99 < 30 and p99 > p1:
        img_infer = np.clip(
            (img_infer.astype(np.float32) - p1) / (p99 - p1) * 255, 0, 255
        ).astype(np.uint8)

    gray_i  = cv2.cvtColor(img_infer, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_i)
    img_infer = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    return img_display, img_infer


def extract_instances(result, size: int = IMG_SIZE):
    """Extrae lista de (centroid_y, binary_mask, confidence) de un resultado YOLO."""
    instances = []
    if result.masks is None:
        return instances
    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
    for idx, m in enumerate(result.masks.data.cpu().numpy()):
        binary = cv2.resize(
            (m > 0.5).astype(np.uint8), (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
        ys, xs = np.where(binary)
        if len(ys) == 0:
            continue
        cy = float(ys.mean())
        cx = float(xs.mean())
        conf = float(confs[idx]) if idx < len(confs) else 0.0
        instances.append((cy, cx, binary, conf))
    instances.sort(key=lambda x: x[0])   # orden top→bottom
    return instances


def iou_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calcula IoU entre dos máscaras binarias."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def nms_instances(
    instances: list[tuple],
    iou_threshold: float = 0.3,
    max_count: int = 17,
) -> list[tuple]:
    """
    Non-Maximum Suppression sobre instancias de máscaras.
    instances: lista de (cy, cx, mask, conf) ordenada por confianza descendente.
    Elimina instancias con IoU > iou_threshold respecto a una ya aceptada.
    Limita el resultado a max_count.
    """
    # Ordenar por confianza descendente
    sorted_inst = sorted(instances, key=lambda x: x[3], reverse=True)
    kept = []
    for candidate in sorted_inst:
        cy, cx, mask, conf = candidate
        # Verificar solapamiento con cada instancia ya aceptada
        suppressed = False
        for _, _, kept_mask, _ in kept:
            if iou_masks(mask, kept_mask) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
        if len(kept) >= max_count:
            break
    # Restaurar orden craniocaudal (Y ascendente)
    kept.sort(key=lambda x: x[0])
    return kept


def assign_labels(instances: list, labels: list[str]) -> list[dict]:
    """Asigna etiquetas vertebrales en orden craniocaudal. Nunca excede len(labels)."""
    results = []
    # Truncar al número de etiquetas disponibles
    for i, (cy, cx, mask, conf) in enumerate(instances[: len(labels)]):
        results.append({
            "label": labels[i],
            "confidence": round(conf, 4),
            "centroid_x": round(cx, 1),
            "centroid_y": round(cy, 1),
            "area_px": int(mask.sum()),
            "mask": mask,
        })
    return results


def ensemble_predict(img_display: np.ndarray, img_infer: np.ndarray, conf: float = 0.25) -> tuple[list[dict], str]:
    """
    Ensemble con NMS y límite estricto de 17 vértebras (T1-T12, L1-L5).

    Estrategia:
      1. Modelo T1-T8  → predice sobre crop superior (42%) → hasta 8 instancias
      2. Modelo completo → predice sobre imagen completa → instancias con cy > CROP_H
      3. NMS global para eliminar duplicados entre modelos
      4. Truncar a máximo 17, asignar etiquetas en orden craniocaudal
    """
    if model_full is None and model_t1t8 is None:
        raise RuntimeError("Ningún modelo está cargado.")

    # Alias corto para legibilidad
    img = img_display  # usado solo en comentarios; inferencia usa img_infer

    all_upper: list[tuple] = []   # (cy, cx, mask, conf) región superior
    all_lower: list[tuple] = []   # región inferior
    mode = "ensemble"

    # ── 1. Modelo T1-T8 sobre crop superior (42% de la imagen) ──────────────
    if model_t1t8 is not None:
        crop        = img_infer[:CROP_H, :]
        crop_scaled = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        res_t1t8    = model_t1t8.predict(
            crop_scaled, conf=conf, verbose=False, device=device
        )[0]
        raw = extract_instances(res_t1t8, IMG_SIZE)

        for cy_crop, cx, mask_crop, cf in raw:
            # Remapear máscara al espacio completo (IMG_SIZE × IMG_SIZE)
            mask_small = cv2.resize(
                mask_crop, (IMG_SIZE, CROP_H), interpolation=cv2.INTER_NEAREST
            )
            mask_full = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            mask_full[:CROP_H, :] = mask_small
            ys2, xs2 = np.where(mask_full)
            if len(ys2) == 0:
                continue
            cy2 = float(ys2.mean())
            cx2 = float(xs2.mean())
            all_upper.append((cy2, cx2, mask_full, cf))

        # NMS en región superior → máximo 8 (T1-T8)
        all_upper = nms_instances(all_upper, iou_threshold=0.30, max_count=8)
    else:
        mode = "full_only"

    # ── 2. Modelo completo → solo instancias debajo del CROP_H ──────────────
    if model_full is not None:
        res_full = model_full.predict(
            img_infer, conf=conf, verbose=False, device=device
        )[0]
        raw_full = extract_instances(res_full, IMG_SIZE)

        # Instancias cuyo centroide está en la región inferior
        lower_raw = [(cy, cx, m, cf) for cy, cx, m, cf in raw_full if cy >= CROP_H]

        # Cuántas vértebras superiores ya tenemos para saber cuántas faltan
        n_upper      = len(all_upper)
        remaining_n  = max(0, 17 - n_upper)   # máximo restante

        all_lower = nms_instances(lower_raw, iou_threshold=0.30, max_count=remaining_n)
    else:
        if model_t1t8 is not None:
            mode = "t1t8_only"

    # ── 3. Combinar y asignar etiquetas ─────────────────────────────────────
    combined = all_upper + all_lower          # ya vienen en orden Y por NMS
    combined.sort(key=lambda x: x[0])        # garantizar orden craniocaudal

    # Asignar etiquetas de VERTEBRA_LABELS en secuencia (máx 17)
    labeled = assign_labels(combined, VERTEBRA_LABELS)

    # Marcar fuente de cada vértebra
    n_upper_final = len(all_upper)
    for i, v in enumerate(labeled):
        v["source"] = "t1t8_model" if i < n_upper_final else "full_model"

    return labeled, mode


def draw_segmentation(img: np.ndarray, vertebrae: list[dict]) -> np.ndarray:
    """
    Dibuja máscaras + etiqueta + confianza sobre la imagen.
    Usa colores brillantes y alto contraste para ser visible sobre radiografías oscuras.
    """
    # Trabajar con copia en float para mezclas precisas
    base   = img.copy()
    result = img.copy().astype(np.float32)

    # Paleta de colores por índice vertebral para mejor distinción visual
    PALETTE_T = [
        (255, 100, 80),   # rojo-naranja
        (255, 160, 60),
        (255, 220, 50),
        (200, 255, 60),
        (100, 255, 100),
        (60,  255, 180),
        (60,  220, 255),
        (80,  140, 255),
        (140,  80, 255),
        (200,  60, 255),
        (255,  60, 200),
        (255,  80, 120),
    ]
    PALETTE_L = [
        (60,  200, 255),
        (100, 240, 255),
        (150, 255, 255),
        (200, 255, 240),
        (220, 255, 200),
    ]

    t_idx = 0
    l_idx = 0

    for v in vertebrae:
        label = v["label"]
        conf  = v["confidence"]
        mask  = v["mask"]

        if label.startswith("T"):
            color = PALETTE_T[t_idx % len(PALETTE_T)]
            t_idx += 1
        else:
            color = PALETTE_L[l_idx % len(PALETTE_L)]
            l_idx += 1

        color_bgr = (color[2], color[1], color[0])  # RGB→BGR

        # Relleno semitransparente (alfa 0.40)
        colored = np.zeros_like(base, dtype=np.float32)
        colored[mask == 1] = color_bgr
        mask3 = np.stack([mask, mask, mask], axis=-1).astype(np.float32)
        result = result * (1 - mask3 * 0.40) + colored * 0.40

        # Contorno grueso (2px) con el mismo color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result.astype(np.uint8), contours, -1, color_bgr, 2)
        # Dibujar también en result como float
        tmp = result.astype(np.uint8)
        cv2.drawContours(tmp, contours, -1, color_bgr, 2)
        result = tmp.astype(np.float32)

        # Etiqueta con fondo sólido negro y texto de color brillante
        cx = int(v["centroid_x"])
        cy = int(v["centroid_y"])
        text = f"{label} {conf:.2f}"
        font_scale = 0.40
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        tx = max(0, min(cx - tw // 2, IMG_SIZE - tw - 2))
        ty = max(th + 2, min(cy, IMG_SIZE - baseline - 2))

        # Fondo negro sólido para máxima legibilidad
        result_u8 = result.astype(np.uint8)
        cv2.rectangle(
            result_u8,
            (tx - 2, ty - th - 2),
            (tx + tw + 2, ty + baseline),
            (0, 0, 0), -1
        )
        cv2.putText(
            result_u8, text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness, cv2.LINE_AA
        )
        result = result_u8.astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


def image_to_base64(img: np.ndarray) -> str:
    """Convierte array BGR → PNG → base64 string."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "models": {
            "full_model": model_full is not None,
            "t1t8_model": model_t1t8 is not None,
        },
        "device": device,
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "healthy": (
            model_full is not None
            or model_t1t8 is not None
            or rbunet_ensemble is not None
        ),
        "full_model_loaded": model_full is not None,
        "t1t8_model_loaded": model_t1t8 is not None,
        "rbunet_loaded": rbunet_ensemble is not None,
        "rbunet_n_folds": getattr(rbunet_ensemble, "n_folds", 0),
        "device": device,
        "img_size": IMG_SIZE,
        "crop_ratio": CROP_RATIO,
    }


@app.post("/segment", response_model=SegmentationResponse, tags=["Segmentation"])
async def segment(
    file: UploadFile = File(..., description="Imagen de radiografía (JPG, PNG, DICOM-PNG)"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Umbral de confianza YOLO (default 0.25, rango recomendado 0.15-0.50)"),
    return_image: bool = Query(True, description="Incluir imagen segmentada en base64"),
):
    """
    Segmenta las vértebras en la radiografía y devuelve:
    - **vertebrae**: lista con etiqueta, confianza, centroide y área por vértebra
    - **image_base64**: imagen con máscaras y etiquetas superpuestas (PNG base64)
    - **model_used**: modo del ensemble empleado
    """
    # Validar tipo de archivo
    if file.content_type not in ("image/jpeg", "image/png", "image/tiff", "image/bmp"):
        # Intentamos de todas formas — YOLO acepta varios formatos
        pass

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        img_display, img_infer = read_image(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo leer la imagen: {e}")

    try:
        vertebrae, mode = ensemble_predict(img_display, img_infer, conf=conf)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Construir respuesta de vértebras (sin la máscara)
    verts_out = [
        VertebraResult(
            label=v["label"],
            confidence=v["confidence"],
            centroid_x=v["centroid_x"],
            centroid_y=v["centroid_y"],
            area_px=v["area_px"],
            source=v["source"],
        )
        for v in vertebrae
    ]

    # Imagen segmentada
    img_b64 = ""
    if return_image:
        segmented = draw_segmentation(img_display, vertebrae)
        img_b64 = image_to_base64(segmented)

    return SegmentationResponse(
        vertebrae=verts_out,
        total_detected=len(verts_out),
        image_base64=img_b64,
        model_used=mode,
        device=device,
    )


@app.post("/segment/full-only", response_model=SegmentationResponse, tags=["Segmentation"])
async def segment_full_only(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    return_image: bool = Query(True),
):
    """Usa **solo** el modelo completo (T1–L5) sin ensemble."""
    if model_full is None:
        raise HTTPException(status_code=503, detail="Modelo completo no cargado.")

    raw_bytes = await file.read()
    img_display, img_infer = read_image(raw_bytes)

    res  = model_full.predict(img_infer, conf=conf, verbose=False, device=device)[0]
    raw  = extract_instances(res, IMG_SIZE)
    raw  = nms_instances(raw, iou_threshold=0.30, max_count=17)
    vertebrae = assign_labels(raw, VERTEBRA_LABELS)
    for v in vertebrae:
        v["source"] = "full_model"

    verts_out = [
        VertebraResult(
            label=v["label"], confidence=v["confidence"],
            centroid_x=v["centroid_x"], centroid_y=v["centroid_y"],
            area_px=v["area_px"], source=v["source"],
        )
        for v in vertebrae
    ]

    img_b64 = ""
    if return_image:
        segmented = draw_segmentation(img_display, vertebrae)
        img_b64   = image_to_base64(segmented)

    return SegmentationResponse(
        vertebrae=verts_out, total_detected=len(verts_out),
        image_base64=img_b64, model_used="full_only", device=device,
    )


@app.post("/segment/t1t8-only", response_model=SegmentationResponse, tags=["Segmentation"])
async def segment_t1t8_only(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    return_image: bool = Query(True),
):
    """Usa **solo** el modelo especializado T1–T8 (crop superior 42%)."""
    if model_t1t8 is None:
        raise HTTPException(status_code=503, detail="Modelo T1-T8 no cargado.")

    raw_bytes = await file.read()
    img_display, img_infer = read_image(raw_bytes)

    crop        = img_infer[:CROP_H, :]
    crop_scaled = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    res         = model_t1t8.predict(crop_scaled, conf=conf, verbose=False, device=device)[0]
    raw         = extract_instances(res, IMG_SIZE)
    raw         = nms_instances(raw, iou_threshold=0.30, max_count=8)

    # Remapear al espacio completo
    remapped = []
    for cy, cx, mask, cf in raw:
        mask_small = cv2.resize(mask, (IMG_SIZE, CROP_H), interpolation=cv2.INTER_NEAREST)
        mask_full = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        mask_full[:CROP_H, :] = mask_small
        ys2, xs2 = np.where(mask_full)
        if len(ys2) == 0:
            continue
        remapped.append((float(ys2.mean()), float(xs2.mean()), mask_full, cf))

    t1t8_labels = [f"T{i}" for i in range(1, 9)]
    vertebrae = assign_labels(remapped, t1t8_labels)
    for v in vertebrae:
        v["source"] = "t1t8_model"

    verts_out = [
        VertebraResult(
            label=v["label"], confidence=v["confidence"],
            centroid_x=v["centroid_x"], centroid_y=v["centroid_y"],
            area_px=v["area_px"], source=v["source"],
        )
        for v in vertebrae
    ]

    img_b64 = ""
    if return_image:
        segmented = draw_segmentation(img_display, vertebrae)
        img_b64 = image_to_base64(segmented)

    return SegmentationResponse(
        vertebrae=verts_out,
        total_detected=len(verts_out),
        image_base64=img_b64,
        model_used="t1t8_only",
        device=device,
    )


# ─── RB-UNet (semantic-segmentation) endpoint ─────────────────────────────────
# Decoupled from YOLO entirely. Uses our project's best model: D2 5-fold
# EncoderUNet (RB-UNet) with hflip TTA, mean-softmax ensemble.

@app.post("/segment/rbunet", response_model=SegmentationResponse, tags=["Segmentation"])
async def segment_rbunet(
    file: UploadFile = File(..., description="Radiografía (JPG, PNG)"),
    return_image: bool = Query(True, description="Incluir PNG segmentado base64"),
):
    """RB-UNet D2 5-fold ensemble (proyecto MaIA/IBIO-SD).

    Semantic segmentation 18-class (bg + T1..L5). Para cada clase de
    primer plano se extrae la componente conexa mayor → un instance dict
    compatible con el schema YOLO. Confianza = media de la probabilidad
    softmax sobre los píxeles del blob.

    Difiere de los endpoints `/segment*`: no usa YOLO, ni CLAHE, ni crop
    T1-T8; el preprocessing es el mismo que el del trainer
    (`ai.training.dataset.preprocess_case`, `roi_crop="off"`) para
    mantener paridad train/inference.
    """
    if rbunet_ensemble is None:
        raise HTTPException(status_code=503, detail="RB-UNet ensemble no cargado.")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        from rbunet import bytes_to_input_tensor, probs_to_instances
        img_display, input_tensor = bytes_to_input_tensor(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo leer la imagen: {e}")

    try:
        probs = rbunet_ensemble.predict_probs(input_tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falló la inferencia RB-UNet: {e}")

    vertebrae = probs_to_instances(probs, out_hw=(IMG_SIZE, IMG_SIZE))

    verts_out = [
        VertebraResult(
            label=v["label"],
            confidence=v["confidence"],
            centroid_x=v["centroid_x"],
            centroid_y=v["centroid_y"],
            area_px=v["area_px"],
            source=v["source"],
        )
        for v in vertebrae
    ]

    img_b64 = ""
    if return_image:
        segmented = draw_segmentation(img_display, vertebrae)
        img_b64 = image_to_base64(segmented)

    return SegmentationResponse(
        vertebrae=verts_out,
        total_detected=len(verts_out),
        image_base64=img_b64,
        model_used=f"rbunet_d2_{rbunet_ensemble.n_folds}fold",
        device=device,
    )
