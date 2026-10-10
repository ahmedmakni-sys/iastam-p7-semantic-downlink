"""Detectors that plug into ``sat7.sahi``: ``detector(crops) -> [[Detection, ...], ...]``.

* ``YoloDetector`` -- the paper's detector: the YOLOv8n trained in WP1 (needs ultralytics,
  torch and the weights; run from ``.venv312``). Every crop is resized to ``imgsz`` by the
  model, which is exactly why a large scene fed whole (B1) loses small ships and SAHI (B2)
  does not.
* ``ClassicDetector`` -- a FALLBACK so the full B0-B4 chain runs on a machine without torch or
  weights. It is the classic pre-filter (``sat7.prefilter``) used as a detector, with the same
  "resize to the network input" step as YOLO and a confidence derived from object contrast.
  Its numbers are not the YOLO numbers and are labelled as such by every script that uses it.
"""
from __future__ import annotations

import cv2
import numpy as np

from .prefilter import PrefilterConfig, run_prefilter
from .sahi import Detection


class YoloDetector:
    name = "yolo"

    def __init__(self, weights: str, imgsz: int = 768, conf: float = 0.05, device=None):
        from ultralytics import YOLO          # lazy: only the 3.12 environment has it
        import torch
        self.model = YOLO(str(weights))
        self.imgsz, self.conf = imgsz, conf
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")

    def __call__(self, crops: list[np.ndarray]) -> list[list[Detection]]:
        res = self.model.predict(crops, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False)
        out = []
        for r in res:
            xywh = r.boxes.xywh.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            out.append([Detection(float(b[0] - b[2] / 2), float(b[1] - b[3] / 2),
                                  float(b[2]), float(b[3]), float(p), int(c))
                        for b, p, c in zip(xywh, conf, cls)])
        return out


class ClassicDetector:
    """Classic CV detector with a network-like input size. FALLBACK, see module docstring."""
    name = "classic (fallback)"

    def __init__(self, input_size: int = 768, cfg: PrefilterConfig | None = None,
                 contrast_full: float = 200.0, conf_floor: float = 0.05):
        self.input_size = input_size
        self.cfg = cfg or PrefilterConfig()
        self.contrast_full = contrast_full      # grey-level contrast that maps to confidence 1
        self.conf_floor = conf_floor

    def __call__(self, crops: list[np.ndarray]) -> list[list[Detection]]:
        return [self._one(c) for c in crops]

    def _one(self, img: np.ndarray) -> list[Detection]:
        h, w = img.shape[:2]
        s = min(1.0, self.input_size / max(h, w))
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) \
            if s < 1.0 else img
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        sea = float(np.median(gray))
        out = []
        for b in run_prefilter(small, self.cfg).boxes:
            patch = gray[b.y:b.y + b.h, b.x:b.x + b.w]
            if patch.size == 0:
                continue
            p = float(np.clip((float(patch.max()) - sea) / self.contrast_full, 0.0, 1.0))
            if p < self.conf_floor:
                continue
            out.append(Detection(b.x / s, b.y / s, b.w / s, b.h / s, p, 0))
        return out
