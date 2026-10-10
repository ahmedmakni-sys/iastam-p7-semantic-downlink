"""Semantic representation and adaptive downlink policy (paper IV, Table I).

A semantic record, eq. (7):

    r = {class, confidence, bbox_global, timestamp, image_id}   (+ lat/lon when georeferenced)

Priority levels, Table I:

    P0  no transmission / discard
    P1  structured semantic metadata
    P2  metadata + compressed ROI
    P3  metadata + ROI + contextual image

The policy turns a detection into a priority. The less sure the detector is, the more visual
evidence the ground receives, so a person (or a ground model) can confirm it:

    p <  t_discard              -> P0
    t_discard <= p < t_context  -> P3  (weak: ROI and a wider context view)
    t_context <= p < t_meta     -> P2  (plausible: ROI)
    p >= t_meta                 -> P1  (confident: metadata alone)

All three thresholds are configurable, as the paper requires ("semantic-priority thresholds").

``SemanticPacket.encode`` is a real binary serialisation, so the transmitted volume D_tx is the
length of actual bytes, not an estimate; ``decode`` is what the ground runs (T_decoding, eq. 28).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import cv2
import numpy as np

from .sahi import Detection

P0, P1, P2, P3 = 0, 1, 2, 3
PRIORITY_NAMES = {P0: "P0 discard", P1: "P1 metadata", P2: "P2 metadata+ROI",
                  P3: "P3 metadata+ROI+context"}

# record: image_id u32 | timestamp f64 (unix s) | x y w h u16 | class u8 | confidence u8
#         | priority u8 | flags u8 (bit0: geo present) [| lat f32 | lon f32]
_REC = struct.Struct("<IdHHHHBBBB")
_GEO = struct.Struct("<ff")
_HDR = struct.Struct("<4sIdH")           # magic, image_id, timestamp, n_records
_BLOB = struct.Struct("<HBI")            # record index, kind (1 ROI, 2 context), length
MAGIC = b"SEM1"


@dataclass
class SemanticRecord:
    """Eq. (7). ``bbox_global`` is (x, y, w, h) in the original image's pixel frame."""
    cls: int
    confidence: float
    bbox_global: tuple[int, int, int, int]
    timestamp: float
    image_id: int
    priority: int = P1
    lat: float | None = None
    lon: float | None = None

    @classmethod
    def from_detection(cls, d: Detection, timestamp: float, image_id: int, priority: int = P1,
                       geo=None) -> "SemanticRecord":
        box = (int(round(d.x)), int(round(d.y)), max(1, int(round(d.w))), max(1, int(round(d.h))))
        lat, lon = geo(d) if geo is not None else (None, None)
        return cls(d.c, d.p, box, timestamp, image_id, priority, lat, lon)

    def pack(self) -> bytes:
        x, y, w, h = (min(max(v, 0), 65535) for v in self.bbox_global)
        geo = self.lat is not None and self.lon is not None
        b = _REC.pack(self.image_id, self.timestamp, x, y, w, h, self.cls,
                      int(round(np.clip(self.confidence, 0, 1) * 255)), self.priority, int(geo))
        return b + (_GEO.pack(self.lat, self.lon) if geo else b"")


@dataclass
class PriorityPolicy:
    """Table I thresholds. ``adaptive=False`` / ``roi=False`` are the paper's ablations."""
    t_discard: float = 0.25
    t_context: float = 0.40
    t_meta: float = 0.70
    adaptive: bool = True        # False: every kept detection gets the same level (fixed_level)
    fixed_level: int = P2
    roi: bool = True             # False: ROIs never sent, P2/P3 degrade to P1

    def priority(self, p: float) -> int:
        if p < self.t_discard:
            return P0
        if not self.adaptive:
            level = self.fixed_level
        elif p < self.t_context:
            level = P3
        elif p < self.t_meta:
            level = P2
        else:
            level = P1
        return level if self.roi else P1


@dataclass
class RoiConfig:
    margin: float = 0.5          # ROI = box grown by this fraction of its size on each side
    min_side: int = 32           # never smaller than this (px)
    jpeg_q: int = 80
    context_scale: float = 6.0   # context view = this many times the box, ...
    context_min: int = 256       # ... at least this wide in the scene ...
    context_px: int = 128        # ... downsampled to this, to show surroundings cheaply
    context_q: int = 60


def _crop(img: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    h, w = img.shape[:2]
    half = side / 2
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    return img[y0:y1, x0:x1]


def _jpeg(img: np.ndarray, q: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return buf.tobytes() if ok else b""


def extract_roi(img: np.ndarray, r: SemanticRecord, cfg: RoiConfig) -> bytes:
    x, y, w, h = r.bbox_global
    side = max(cfg.min_side, max(w, h) * (1 + 2 * cfg.margin))
    return _jpeg(_crop(img, x + w / 2, y + h / 2, side), cfg.jpeg_q)


def extract_context(img: np.ndarray, r: SemanticRecord, cfg: RoiConfig) -> bytes:
    x, y, w, h = r.bbox_global
    side = max(cfg.context_min, max(w, h) * cfg.context_scale)
    view = _crop(img, x + w / 2, y + h / 2, side)
    if view.size == 0:
        return b""
    view = cv2.resize(view, (cfg.context_px, cfg.context_px), interpolation=cv2.INTER_AREA)
    return _jpeg(view, cfg.context_q)


@dataclass
class SemanticPacket:
    image_id: int
    timestamp: float
    records: list[SemanticRecord] = field(default_factory=list)
    blobs: list[tuple[int, int, bytes]] = field(default_factory=list)   # (record idx, kind, bytes)

    def encode(self) -> bytes:
        out = [_HDR.pack(MAGIC, self.image_id, self.timestamp, len(self.records))]
        out += [r.pack() for r in self.records]
        out.append(struct.pack("<H", len(self.blobs)))
        for i, kind, b in self.blobs:
            out.append(_BLOB.pack(i, kind, len(b)))
            out.append(b)
        return b"".join(out)

    @classmethod
    def decode(cls, data: bytes) -> "SemanticPacket":
        magic, image_id, ts, n = _HDR.unpack_from(data, 0)
        if magic != MAGIC:
            raise ValueError("not a semantic packet")
        off = _HDR.size
        recs = []
        for _ in range(n):
            iid, t, x, y, w, h, c, q, pr, flags = _REC.unpack_from(data, off)
            off += _REC.size
            lat = lon = None
            if flags & 1:
                lat, lon = _GEO.unpack_from(data, off)
                off += _GEO.size
            recs.append(SemanticRecord(c, q / 255, (x, y, w, h), t, iid, pr, lat, lon))
        (nb,) = struct.unpack_from("<H", data, off)
        off += 2
        blobs = []
        for _ in range(nb):
            i, kind, ln = _BLOB.unpack_from(data, off)
            off += _BLOB.size
            blobs.append((i, kind, data[off:off + ln]))
            off += ln
        return cls(image_id, ts, recs, blobs)


def build_packet(img: np.ndarray, dets: list[Detection], image_id: int, timestamp: float,
                 policy: PriorityPolicy | None = None, roi: RoiConfig | None = None,
                 geo=None, semantic: bool = True) -> SemanticPacket:
    """Detections -> one semantic packet.

    ``semantic=False`` is the paper's "detection output" (configurations B1/B2): every
    detection above threshold as a metadata record, no policy, no image evidence.
    """
    policy, roi = policy or PriorityPolicy(), roi or RoiConfig()
    pkt = SemanticPacket(image_id, timestamp)
    for d in dets:
        level = policy.priority(d.p) if semantic else P1
        if level == P0:
            continue
        r = SemanticRecord.from_detection(d, timestamp, image_id, level, geo)
        pkt.records.append(r)
        i = len(pkt.records) - 1
        if level >= P2:
            pkt.blobs.append((i, 1, extract_roi(img, r, roi)))
        if level >= P3:
            pkt.blobs.append((i, 2, extract_context(img, r, roi)))
    return pkt
