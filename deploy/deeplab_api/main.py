from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
import segmentation_models_pytorch as smp


# ============================================================
# Basic config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BINARY_MODEL_PATH = BASE_DIR / "models" / "best_model_binary.pth"
MULTI_MODEL_PATH = BASE_DIR / "models" / "best_model_multi.pth"

# Fallback values. If the .pth contains cfg, the API will try to use that.
BINARY_IMAGE_SIZE = int(os.getenv("BINARY_IMAGE_SIZE", "768"))
MULTI_IMAGE_SIZE = int(os.getenv("MULTI_IMAGE_SIZE", "1024"))

BINARY_ENCODER = os.getenv("BINARY_ENCODER", "efficientnet-b4")
MULTI_ENCODER = os.getenv("MULTI_ENCODER", "timm-efficientnet-b5")

BINARY_THRESHOLD = float(os.getenv("BINARY_THRESHOLD", "0.6"))

MIN_COMPONENT_AREA = int(os.getenv("MIN_COMPONENT_AREA", "30"))
POLYGON_EPSILON_RATIO = float(os.getenv("POLYGON_EPSILON_RATIO", "0.004"))

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CLASS_LABELS = {
    0: "background",
    1: "T1",
    2: "T2",
    3: "T3",
    4: "T4",
    5: "T5",
    6: "T6",
    7: "T7",
    8: "T8",
    9: "T9",
    10: "T10",
    11: "T11",
    12: "T12",
    13: "L1",
    14: "L2",
    15: "L3",
    16: "L4",
    17: "L5",
}


app = FastAPI(title="Scoliosis DeepLabV3+ API")


# ============================================================
# Utility functions
# ============================================================

def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def get_cfg_value(cfg, key: str, default):
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    return getattr(cfg, key, default)


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    return cleaned


def build_deeplab_model(
    checkpoint_path: Path,
    fallback_encoder: str,
    fallback_image_size: int,
    classes: int,
):
    checkpoint = load_checkpoint(checkpoint_path)

    if isinstance(checkpoint, dict):
        cfg = checkpoint.get("cfg")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        cfg = None
        state_dict = checkpoint

    encoder_name = get_cfg_value(cfg, "ENCODER_NAME", fallback_encoder)
    image_size = int(get_cfg_value(cfg, "IMAGE_SIZE", fallback_image_size))

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=classes,
    )

    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.to(DEVICE)
    model.eval()

    return model, image_size, encoder_name


def read_upload_as_rgb(upload: UploadFile) -> np.ndarray:
    image = Image.open(upload.file).convert("RGB")
    return np.array(image, dtype=np.uint8)


def resize_pad_image(image_rgb: np.ndarray, image_size: int):
    """
    Resize preserving aspect ratio, then pad to square.

    Returns:
      padded image
      metadata needed to undo the padding
    """
    orig_h, orig_w = image_rgb.shape[:2]

    scale = image_size / max(orig_h, orig_w)
    resized_h = int(round(orig_h * scale))
    resized_w = int(round(orig_w * scale))

    resized = cv2.resize(
        image_rgb,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_top = (image_size - resized_h) // 2
    pad_bottom = image_size - resized_h - pad_top
    pad_left = (image_size - resized_w) // 2
    pad_right = image_size - resized_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "image_size": image_size,
    }

    return padded, meta


def normalize_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """
    ImageNet normalization, matching your notebook transform.
    """
    image = image_rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))

    tensor = torch.from_numpy(image).unsqueeze(0).float()
    return tensor.to(DEVICE)


def restore_mask_to_original_size(mask_canvas: np.ndarray, meta: Dict) -> np.ndarray:
    """
    Removes padding and resizes the prediction back to the original image size.
    This is what keeps frontend polygons aligned to the original image.
    """
    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]
    resized_h = meta["resized_h"]
    resized_w = meta["resized_w"]
    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]

    unpadded = mask_canvas[
        pad_top:pad_top + resized_h,
        pad_left:pad_left + resized_w,
    ]

    restored = cv2.resize(
        unpadded,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return restored


def restore_prob_to_original_size(prob_canvas: np.ndarray, meta: Dict) -> np.ndarray:
    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]
    resized_h = meta["resized_h"]
    resized_w = meta["resized_w"]
    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]

    unpadded = prob_canvas[
        pad_top:pad_top + resized_h,
        pad_left:pad_left + resized_w,
    ]

    restored = cv2.resize(
        unpadded,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )

    return restored


def contour_to_polygon(contour: np.ndarray) -> List[List[int]]:
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, POLYGON_EPSILON_RATIO * perimeter)

    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2)

    return [[int(x), int(y)] for x, y in points]


def mask_to_segments(
    mask: np.ndarray,
    label_map: Dict[int, str],
    confidence_map: Optional[np.ndarray] = None,
    allowed_ids: Optional[List[int]] = None,
):
    segments = []

    if allowed_ids is None:
        ids = sorted([int(x) for x in np.unique(mask) if int(x) != 0])
    else:
        ids = allowed_ids

    segment_id = 1

    for class_id in ids:
        if class_id == 0:
            continue

        class_mask = (mask == class_id).astype(np.uint8)

        if class_mask.sum() < MIN_COMPONENT_AREA:
            continue

        contours, _ = cv2.findContours(
            class_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            pixel_area = int(cv2.contourArea(contour))

            if pixel_area < MIN_COMPONENT_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            polygon = contour_to_polygon(contour)

            if len(polygon) < 3:
                continue

            confidence = None
            if confidence_map is not None:
                component_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(component_mask, [contour], -1, 1, thickness=-1)

                values = confidence_map[component_mask == 1]
                if values.size > 0:
                    confidence = float(np.mean(values))

            item = {
                "id": segment_id,
                "class_id": int(class_id),
                "label": label_map.get(int(class_id), f"class_{class_id}"),
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(class_mask.sum()),
                "polygon": polygon,
            }

            if confidence is not None:
                item["confidence"] = round(confidence, 4)

            segments.append(item)
            segment_id += 1

    return segments


# ============================================================
# Load models once at startup
# ============================================================

binary_model = None
multi_model = None

binary_size = None
multi_size = None

binary_encoder = None
multi_encoder = None


@app.on_event("startup")
def startup_event():
    global binary_model, multi_model
    global binary_size, multi_size
    global binary_encoder, multi_encoder

    print("Using device:", DEVICE)

    print("Loading binary model...")
    binary_model, binary_size, binary_encoder = build_deeplab_model(
        checkpoint_path=BINARY_MODEL_PATH,
        fallback_encoder=BINARY_ENCODER,
        fallback_image_size=BINARY_IMAGE_SIZE,
        classes=1,
    )
    print(f"Binary model loaded: encoder={binary_encoder}, image_size={binary_size}")

    print("Loading multiclass model...")
    multi_model, multi_size, multi_encoder = build_deeplab_model(
        checkpoint_path=MULTI_MODEL_PATH,
        fallback_encoder=MULTI_ENCODER,
        fallback_image_size=MULTI_IMAGE_SIZE,
        classes=18,
    )
    print(f"Multiclass model loaded: encoder={multi_encoder}, image_size={multi_size}")


# ============================================================
# Prediction functions
# ============================================================

@torch.inference_mode()
def predict_binary(image_rgb: np.ndarray):
    padded, meta = resize_pad_image(image_rgb, binary_size)
    tensor = normalize_to_tensor(padded)

    logits = binary_model(tensor)

    prob_canvas = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask_canvas = (prob_canvas >= BINARY_THRESHOLD).astype(np.uint8)

    prob_original = restore_prob_to_original_size(prob_canvas, meta)
    mask_original = restore_mask_to_original_size(mask_canvas, meta)

    segments = mask_to_segments(
        mask=mask_original,
        label_map={1: "spine"},
        confidence_map=prob_original,
        allowed_ids=[1],
    )

    return {
        "type": "binary",
        "model": {
            "architecture": "DeepLabV3Plus",
            "encoder": binary_encoder,
            "image_size": binary_size,
            "threshold": BINARY_THRESHOLD,
        },
        "segments": segments,
    }


@torch.inference_mode()
def predict_multiclass(image_rgb: np.ndarray):
    padded, meta = resize_pad_image(image_rgb, multi_size)
    tensor = normalize_to_tensor(padded)

    logits = multi_model(tensor)

    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    mask_canvas = np.argmax(probs, axis=0).astype(np.uint8)
    confidence_canvas = np.max(probs, axis=0).astype(np.float32)

    mask_original = restore_mask_to_original_size(mask_canvas, meta)
    confidence_original = restore_prob_to_original_size(confidence_canvas, meta)

    segments = mask_to_segments(
        mask=mask_original,
        label_map=CLASS_LABELS,
        confidence_map=confidence_original,
        allowed_ids=list(range(1, 18)),
    )

    classes_detected = sorted(
        list({segment["label"] for segment in segments}),
        key=lambda label: list(CLASS_LABELS.values()).index(label)
        if label in CLASS_LABELS.values()
        else 999,
    )

    return {
        "type": "multiclass",
        "model": {
            "architecture": "DeepLabV3Plus",
            "encoder": multi_encoder,
            "image_size": multi_size,
            "classes": 18,
        },
        "classes_detected": classes_detected,
        "segments": segments,
    }


# ============================================================
# API endpoints
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "binary_model_loaded": binary_model is not None,
        "multi_model_loaded": multi_model is not None,
        "binary_size": binary_size,
        "multi_size": multi_size,
    }


@app.post("/predict-binary")
async def predict_binary_endpoint(file: UploadFile = File(...)):
    image_rgb = read_upload_as_rgb(file)
    h, w = image_rgb.shape[:2]

    binary_result = predict_binary(image_rgb)

    return {
        "image_width": int(w),
        "image_height": int(h),
        "results": {
            "binary": binary_result,
        },
    }


@app.post("/predict-multiclass")
async def predict_multiclass_endpoint(file: UploadFile = File(...)):
    image_rgb = read_upload_as_rgb(file)
    h, w = image_rgb.shape[:2]

    multiclass_result = predict_multiclass(image_rgb)

    return {
        "image_width": int(w),
        "image_height": int(h),
        "results": {
            "multiclass": multiclass_result,
        },
    }


@app.post("/predict-full")
async def predict_full_endpoint(file: UploadFile = File(...)):
    image_rgb = read_upload_as_rgb(file)
    h, w = image_rgb.shape[:2]

    binary_result = predict_binary(image_rgb)
    multiclass_result = predict_multiclass(image_rgb)

    return {
        "image_width": int(w),
        "image_height": int(h),
        "results": {
            "binary": binary_result,
            "multiclass": multiclass_result,
        },
    }