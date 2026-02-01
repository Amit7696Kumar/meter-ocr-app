from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ✅ DO NOT import ultralytics / torch / easyocr at top-level.
# They MUST be lazy-loaded inside functions to prevent server import crash/hang.

def log(msg: str) -> None:
    print(msg, flush=True, file=sys.stdout)


BASE_DIR = os.path.dirname(__file__)

YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "lcd_best.pt"),
)

DEBUG_DIR = Path(os.path.join(BASE_DIR, "static", "debug"))
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

_yolo = None
_reader = None


def _clean_digits(s: str) -> str:
    if not s:
        return ""
    s = s.strip()

    trans = str.maketrans({
        "O": "0", "o": "0",
        "I": "1", "l": "1", "|": "1",
        "S": "5", "s": "5",
        "B": "8",
        "Z": "2",
        "G": "6",
        "q": "9",
    })
    s = s.translate(trans)
    s = re.sub(r"[^0-9.]", "", s)

    if s.count(".") > 1:
        first = s.find(".")
        s = s[:first + 1] + s[first + 1:].replace(".", "")

    return s


def _best_number_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return None
    nums.sort(key=lambda x: len(x.replace(".", "")), reverse=True)
    return nums[0]


def _pad_box(x0: int, y0: int, x1: int, y1: int, w: int, h: int, pad: float = 0.10) -> Tuple[int, int, int, int]:
    bw = x1 - x0
    bh = y1 - y0
    px = int(bw * pad)
    py = int(bh * pad)
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(w - 1, x1 + px)
    y1 = min(h - 1, y1 + py)
    return x0, y0, x1, y1


def get_yolo():
    """
    Lazy-load YOLO ONLY when needed.
    Also disables TorchDynamo/Inductor to avoid torch._dynamo + sympy slow import paths.
    """
    global _yolo
    if _yolo is not None:
        return _yolo

    # ✅ Disable dynamo/compile paths (huge stability improvement on mac)
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")

    log(f"[YOLO] Loading model: {YOLO_MODEL_PATH}")

    # ✅ Import ultralytics HERE (not at top-level)
    from ultralytics import YOLO  # noqa

    t0 = time.time()
    _yolo = YOLO(YOLO_MODEL_PATH)
    log(f"[YOLO] Loaded in {time.time() - t0:.2f}s")

    return _yolo


def get_reader():
    global _reader
    if _reader is None:
        log("[OCR] Initializing EasyOCR reader (en)")
        import easyocr  # noqa
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def save_yolo_debug(img_bgr: np.ndarray, box: Tuple[int, int, int, int], out_path: str) -> None:
    x0, y0, x1, y1 = box
    dbg = img_bgr.copy()
    cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.putText(dbg, "LCD", (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(out_path, dbg)


def detect_lcd(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    model = get_yolo()
    results = model.predict(img_bgr, imgsz=640, conf=0.25, verbose=False)
    if not results:
        return None

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), dtype=float)
    best_i = int(np.argmax(confs)) if len(confs) else 0

    x0, y0, x1, y1 = xyxy[best_i].tolist()
    conf = float(confs[best_i]) if len(confs) else 0.0

    h, w = img_bgr.shape[:2]
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))

    if x1 <= x0 or y1 <= y0:
        return None

    x0, y0, x1, y1 = _pad_box(x0, y0, x1, y1, w, h, pad=0.10)
    return x0, y0, x1, y1, conf


def build_variants(crop_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = []
    variants.append(("raw", crop_bgr))

    up = cv2.resize(crop_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    variants.append(("up_raw", up))

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    variants.append(("up_gray", gray))

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)
    variants.append(("up_gray_clahe", g2))

    thr = cv2.adaptiveThreshold(
        g2, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )
    variants.append(("thr", thr))
    variants.append(("thr_inv", cv2.bitwise_not(thr)))
    return variants


def easyocr_digits(img_any: np.ndarray) -> Tuple[str, float]:
    reader = get_reader()

    if img_any.ndim == 3:
        img_rgb = cv2.cvtColor(img_any, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_any

    results = reader.readtext(
        img_rgb,
        detail=1,
        allowlist="0123456789.",
        paragraph=False
    )

    if not results:
        return "", 0.0

    best = max(results, key=lambda r: float(r[2]) if len(r) >= 3 else 0.0)
    text = str(best[1]).strip()
    conf = float(best[2]) if len(best) >= 3 else 0.0
    return text, conf


def warmup_models() -> None:
    # optional: call once at startup if you want
    try:
        get_yolo()
    except Exception as e:
        log(f"[WARMUP] ⚠️ YOLO warmup failed: {e}")


def run_ocr(image_path: str, debug_id: Optional[str] = None) -> Dict[str, Any]:
    log(f"[OCR] Processing image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        return {"numeric": None, "text": "", "lines": [], "used_crop": False, "crop_box": None, "debug": {"error": "cv2.imread failed"}}

    det = detect_lcd(img)
    if det is None:
        log("[YOLO] ❌ No LCD detected")
        return {"numeric": None, "text": "", "lines": [], "used_crop": False, "crop_box": None, "debug": {"yolo": "no box"}}

    x0, y0, x1, y1, yconf = det
    log(f"[YOLO] LCD detected at ({x0},{y0},{x1},{y1}) conf={yconf:.2f}")

    crop = img[y0:y1, x0:x1].copy()

    debug_urls = {}
    if debug_id:
        yolo_path = DEBUG_DIR / f"yolo_{debug_id}.jpg"
        crop_path = DEBUG_DIR / f"crop_{debug_id}.jpg"

        save_yolo_debug(img, (x0, y0, x1, y1), str(yolo_path))
        cv2.imwrite(str(crop_path), crop)

        debug_urls = {
            "yolo": f"/static/debug/yolo_{debug_id}.jpg",
            "crop": f"/static/debug/crop_{debug_id}.jpg",
        }

    variants = build_variants(crop)

    best_value: Optional[str] = None
    best_score: float = -1.0
    best_raw: str = ""
    best_variant: str = ""

    for name, vimg in variants:
        log(f"[OCR] Trying variant: {name}")
        raw, conf = easyocr_digits(vimg)
        if not raw:
            log("[OCR]   (no text)")
            continue

        cleaned = _clean_digits(raw)
        candidate = _best_number_from_text(cleaned)
        log(f"[OCR]   raw='{raw}' conf={conf:.3f} cleaned='{cleaned}' candidate='{candidate}'")

        if candidate:
            length = len(candidate.replace(".", ""))
            score = conf + (0.15 * length)
            if score > best_score:
                best_score = score
                best_value = candidate
                best_raw = raw
                best_variant = name

    if not best_value:
        log("[OCR] ❌ No numeric value extracted")
        return {
            "numeric": None,
            "text": "",
            "lines": [],
            "used_crop": True,
            "crop_box": [x0, y0, x1 - x0, y1 - y0],
            "debug_urls": debug_urls,
            "debug": {"yolo_conf": yconf},
        }

    log(f"[OCR] ✅ FINAL VALUE = {best_value} (variant={best_variant}, score={best_score:.3f})")

    return {
        "numeric": {"value": best_value, "confidence": float(yconf)},
        "text": best_value,
        "lines": [{"text": best_value, "confidence": float(yconf)}],
        "used_crop": True,
        "crop_box": [x0, y0, x1 - x0, y1 - y0],
        "debug_urls": debug_urls,
        "debug": {"yolo_conf": yconf, "best_variant": best_variant, "best_raw": best_raw},
    }