"""The paper's onboard chain as one object (Fig. 1 and 2), stage by stage as in eq. (9).

    large image -> preprocessing -> tiling -> detector per tile -> global coordinates
      (eq. 1)       E_pre           E_tile    E_det (SAHI, eq. 3-4)  (eq. 5)
                -> fusion -> semantic records + priorities -> packet bytes
                   E_fusion   E_semantic (eq. 7, Table I)

``OnboardPipeline.run`` returns the packet that leaves the satellite and the measured time of
every stage, so the energy model (``sat7.energy``) can turn them into E_proc. The ground side
is ``ground_decode``. Configurations B1-B4 and every ablation of the paper are just different
``PipelineConfig`` values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .sahi import Detection, SahiConfig, SahiResult, direct_detect, fuse, sahi_detect
from .semantic import PriorityPolicy, RoiConfig, SemanticPacket, build_packet


@dataclass
class PipelineConfig:
    sahi: bool = True                  # False: the detector on the original image (B1)
    slicing: SahiConfig = field(default_factory=SahiConfig)
    semantic: bool = True              # False: plain "detection output", no policy (B1, B2)
    policy: PriorityPolicy = field(default_factory=PriorityPolicy)
    roi: RoiConfig = field(default_factory=RoiConfig)
    det_ms: float | None = None        # REAL ms per detector inference (energy.detector_timing);
                                       # None: use this machine's measured T_det

    @property
    def detection_key(self) -> tuple:
        """Configs with the same key run the detector on the same crops and can share it."""
        return ("sahi", self.slicing.tile, self.slicing.overlap) if self.sahi else ("direct",)


@dataclass
class OnboardOutput:
    image_id: int
    detections: list[Detection]        # D_global after fusion and the confidence threshold
    packet: bytes                      # what is transmitted: D_tx = len(packet)
    times: dict                        # seconds per stage: pre, tile, det, fusion, semantic

    @property
    def t_inference(self) -> float:
        return sum(v for k, v in self.times.items() if k != "semantic")

    @property
    def t_encoding(self) -> float:
        return self.times.get("semantic", 0.0)


def preprocess(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Paper III-B preprocessing: a contiguous 3-channel uint8 array. Returns (image, T_pre)."""
    t0 = time.perf_counter()
    img = np.ascontiguousarray(image[..., :3] if image.ndim == 3 else np.dstack([image] * 3))
    return img.astype(np.uint8, copy=False), time.perf_counter() - t0


class OnboardPipeline:
    def __init__(self, detector, cfg: PipelineConfig | None = None):
        self.detector = detector
        self.cfg = cfg or PipelineConfig()

    def detect(self, image: np.ndarray) -> SahiResult:
        """Run the detector (sliced or not). The result can be reused by ``run``."""
        if self.cfg.sahi:
            return sahi_detect(image, self.detector, self.cfg.slicing)
        return direct_detect(image, self.detector, self.cfg.slicing.conf_thr)

    def detections(self, detected: SahiResult, conf_thr: float | None = None) -> tuple[list, float]:
        """Threshold + fusion of cached raw detections under this config. Returns (D, T_fusion)."""
        s = self.cfg.slicing
        thr = s.conf_thr if conf_thr is None else conf_thr
        kept = [d for d in detected.raw if d.p >= thr]
        if not self.cfg.sahi:                       # one inference: nothing to fuse
            return kept, 0.0
        t0 = time.perf_counter()
        out = fuse(kept, s.fusion, s.match_metric, s.match_thr)
        return out, time.perf_counter() - t0

    def run(self, image: np.ndarray, image_id: int, timestamp: float,
            detected: SahiResult | None = None) -> OnboardOutput:
        img, t_pre = preprocess(image)
        if detected is None:
            detected = self.detect(img)
        dets, t_fusion = self.detections(detected)
        t0 = time.perf_counter()
        pkt = build_packet(img, dets, image_id, timestamp, self.cfg.policy, self.cfg.roi,
                           semantic=self.cfg.semantic)
        data = pkt.encode()
        t_det = detected.t_det_s
        if self.cfg.det_ms is not None:                # E_det priced on a real detector time
            n_inf = detected.n_slices + int(self.cfg.sahi and self.cfg.slicing.full_pass)
            t_det = n_inf * self.cfg.det_ms / 1000.0
        times = {"pre": t_pre, "tile": detected.t_tile_s, "det": t_det,
                 "fusion": t_fusion, "semantic": time.perf_counter() - t0}
        return OnboardOutput(image_id, dets, data, times)

    def with_threshold(self, t: float) -> "OnboardPipeline":
        """Same pipeline, another confidence threshold and priority ladder (trade-off sweeps)."""
        pol = replace(self.cfg.policy, t_discard=t, t_context=max(t, self.cfg.policy.t_context),
                      t_meta=max(t, self.cfg.policy.t_meta))
        return OnboardPipeline(self.detector, replace(self.cfg, policy=pol,
                                                      slicing=replace(self.cfg.slicing, conf_thr=t)))


def ground_decode(data: bytes) -> tuple[SemanticPacket, float]:
    """Ground station: packet bytes -> records. Returns (packet, T_decoding) for eq. (28)."""
    t0 = time.perf_counter()
    pkt = SemanticPacket.decode(data)
    return pkt, time.perf_counter() - t0


def record_centres(pkt: SemanticPacket) -> list[tuple[int, float, float]]:
    """(class, cx, cy) of every delivered record, for information preservation."""
    return [(r.cls, r.bbox_global[0] + r.bbox_global[2] / 2, r.bbox_global[1] + r.bbox_global[3] / 2)
            for r in pkt.records]
