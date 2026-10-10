"""Large-image tiling, SAHI sliced inference, global coordinates and fusion (paper III-B..D).

This is the onboard perception front end exactly as the paper describes it:

    I in R^{H x W x C}                                  (1)  the large scene
    I -> {I_1, ..., I_N}, neighbouring tiles overlap    (2)  ``slice_grid``
    D_j = {d_j1, ..., d_jm}                             (3)  the detector on each tile
    d = (x, y, w, h, p, c)                              (4)  ``Detection``
    x_g = x_0 + x_l,  y_g = y_0 + y_l                   (5)  ``Detection.to_global``
    D_global = Fuse(D_1, ..., D_N)                      (6)  ``fuse``

Default configuration is the paper's "representative initial configuration": 512 x 512 tiles,
20% overlap. Both are parameters, as the paper requires. Fusion offers both strategies the
paper names -- overlap-based *suppression* (NMS) and *merging* (NMM, what the SAHI library
calls GREEDYNMM) -- with IoU or IoS (intersection over the smaller box) as the overlap measure.
IoS is the SAHI default because a ship cut by a slice border yields a *partial* box that sits
inside the full one: its IoU with the full box is small, its IoS is close to 1.

The detector is any callable ``detector(crops) -> [[Detection, ...], ...]`` returning boxes in
each crop's own (local) coordinates; see ``sat7.detectors``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Detection:
    """d = (x, y, w, h, p, c), eq. (4). (x, y) is the top-left corner, in pixels."""
    x: float
    y: float
    w: float
    h: float
    p: float          # confidence score
    c: int = 0        # predicted class (0 = ship)

    def to_global(self, x0: float, y0: float) -> "Detection":
        """Eq. (5): a tile-local box -> the coordinate system of the original image."""
        return replace(self, x=self.x + x0, y=self.y + y0)

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def inter(self, o: "Detection") -> float:
        iw = min(self.x + self.w, o.x + o.w) - max(self.x, o.x)
        ih = min(self.y + self.h, o.y + o.h) - max(self.y, o.y)
        return max(iw, 0.0) * max(ih, 0.0)

    def iou(self, o: "Detection") -> float:
        i = self.inter(o)
        u = self.area + o.area - i
        return i / u if u > 0 else 0.0

    def ios(self, o: "Detection") -> float:
        """Intersection over the smaller box: catches a slice-cut partial inside a full box."""
        s = min(self.area, o.area)
        return self.inter(o) / s if s > 0 else 0.0


@dataclass
class SahiConfig:
    tile: int = 512                 # paper III-B: 512 x 512 ...
    overlap: float = 0.20           # ... with approximately 20% overlap
    fusion: str = "nmm"             # "nms" (suppression) | "nmm" (merging) | "none"
    match_metric: str = "ios"       # "iou" | "ios"
    match_thr: float = 0.5
    full_pass: bool = False         # SAHI's optional extra inference on the whole (resized) image
    conf_thr: float = 0.25          # detector confidence threshold (paper VIII-D, configurable)


def _starts(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:          # last tile flush with the border, never past it
        s.append(length - tile)
    return s


def slice_grid(height: int, width: int, tile: int = 512, overlap: float = 0.2):
    """Eq. (2): overlapping tiles covering the whole image, as (x0, y0, w, h)."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    tw, th = min(tile, width), min(tile, height)
    return [(x, y, tw, th) for y in _starts(height, tile, overlap)
            for x in _starts(width, tile, overlap)]


def fuse(dets: list[Detection], mode: str = "nmm", metric: str = "ios",
         thr: float = 0.5) -> list[Detection]:
    """Eq. (6): combine detections from overlapping tiles into one global set.

    Greedy, highest confidence first, class-aware. ``nms`` drops a lower-scored box that
    overlaps a kept one; ``nmm`` merges it into the kept box (union of extents, max score), so
    a ship split across two slices comes back as one whole-ship box.
    """
    if mode == "none":
        return list(dets)
    if mode not in ("nms", "nmm"):
        raise ValueError(f"unknown fusion mode {mode!r}")
    overlap = Detection.iou if metric == "iou" else Detection.ios
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda z: -z.p):
        hit = None
        for k, kd in enumerate(kept):
            if kd.c == d.c and overlap(kd, d) >= thr:
                hit = k
                break
        if hit is None:
            kept.append(d)
        elif mode == "nmm":
            kd = kept[hit]
            x0, y0 = min(kd.x, d.x), min(kd.y, d.y)
            x1, y1 = max(kd.x + kd.w, d.x + d.w), max(kd.y + kd.h, d.y + d.h)
            kept[hit] = Detection(x0, y0, x1 - x0, y1 - y0, max(kd.p, d.p), kd.c)
    return kept


@dataclass
class SahiResult:
    detections: list[Detection]                     # D_global, after fusion
    raw: list[Detection] = field(default_factory=list)   # all global boxes before fusion
    n_slices: int = 0
    t_tile_s: float = 0.0                           # T_tile   (eq. 9 stage times)
    t_det_s: float = 0.0                            # T_det
    t_fusion_s: float = 0.0                         # T_fusion


def sahi_detect(image: np.ndarray, detector, cfg: SahiConfig | None = None) -> SahiResult:
    """Paper Fig. 1/2: tiles -> detector per tile -> global coordinates -> fusion."""
    cfg = cfg or SahiConfig()
    h, w = image.shape[:2]
    slices = slice_grid(h, w, cfg.tile, cfg.overlap)
    t0 = time.perf_counter()
    crops = [image[y:y + th, x:x + tw] for (x, y, tw, th) in slices]
    t_tile = time.perf_counter() - t0
    t0 = time.perf_counter()
    local = detector(crops)
    raw = [d.to_global(x, y) for (x, y, _, _), ds in zip(slices, local) for d in ds]  # eq. (5)
    if cfg.full_pass:
        raw += detector([image])[0]
    t_det = time.perf_counter() - t0
    kept = [d for d in raw if d.p >= cfg.conf_thr]
    t0 = time.perf_counter()
    out = fuse(kept, cfg.fusion, cfg.match_metric, cfg.match_thr)
    return SahiResult(out, raw, len(slices), t_tile, t_det, time.perf_counter() - t0)


def direct_detect(image: np.ndarray, detector, conf_thr: float = 0.25):
    """Configuration B1: one inference on the original image (the detector resizes it)."""
    t0 = time.perf_counter()
    raw = detector([image])[0]
    t_det = time.perf_counter() - t0
    return SahiResult([d for d in raw if d.p >= conf_thr], raw, 1, 0.0, t_det, 0.0)
