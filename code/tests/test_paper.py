"""The paper's equations, pinned: tiling (2), global coordinates (5), fusion (6), records (7),
Table I, energy (8)-(17), DR (27), latency (28)-(29), path choice (16), (25)-(26), metrics."""

import numpy as np
import pytest

from sat7.energy import (ObjectiveConfig, PowerConfig, comm_energy_direct, comm_energy_relay,
                         downlink_reduction, energy_saving, processing_energy,
                         select_operating_point, total_latency, transmission_time)
from sat7.metrics import DetAccumulator, preserved
from sat7.relay import LinkSchedule, Network, PathPolicy, Window
from sat7.sahi import Detection, SahiConfig, fuse, sahi_detect, slice_grid
from sat7.semantic import (P0, P1, P2, P3, PriorityPolicy, SemanticPacket, SemanticRecord,
                           build_packet)


# ---------------- (2) tiling
def test_slices_cover_image_with_requested_overlap():
    s = slice_grid(2048, 2048, 512, 0.2)
    cover = np.zeros((2048, 2048), bool)
    for x, y, w, h in s:
        assert (w, h) == (512, 512) and x + w <= 2048 and y + h <= 2048
        cover[y:y + h, x:x + w] = True
    assert cover.all()
    xs = sorted({x for x, *_ in s})
    assert xs[1] - xs[0] == round(512 * 0.8)          # stride = tile * (1 - overlap)


def test_small_image_is_one_slice():
    assert slice_grid(300, 400, 512, 0.2) == [(0, 0, 400, 300)]


# ---------------- (5) global coordinates
def test_to_global_is_a_translation():
    d = Detection(10, 20, 5, 6, 0.9).to_global(512, 410)
    assert (d.x, d.y, d.w, d.h, d.p) == (522, 430, 5, 6, 0.9)


# ---------------- (6) fusion
def test_fusion_merges_a_ship_cut_by_a_slice_border():
    full = Detection(100, 100, 40, 10, 0.9)
    part = Detection(100, 100, 18, 10, 0.6)            # same ship, cut: IoU small, IoS = 1
    assert full.iou(part) < 0.5 and full.ios(part) == pytest.approx(1.0)
    assert len(fuse([full, part], "nms", "ios")) == 1
    assert len(fuse([full, part], "nms", "iou")) == 2
    m = fuse([part, Detection(110, 100, 30, 10, 0.8)], "nmm", "ios", 0.3)
    assert len(m) == 1 and (m[0].x, m[0].w, m[0].p) == (100, 40, 0.8)
    assert len(fuse([full, part], "none")) == 2


def test_fusion_is_class_aware():
    a, b = Detection(0, 0, 10, 10, 0.9, 0), Detection(0, 0, 10, 10, 0.8, 1)
    assert len(fuse([a, b], "nms")) == 2


def test_sahi_returns_global_boxes_and_dedups_overlap():
    img = np.zeros((1024, 1024, 3), np.uint8)
    ship = (600, 300, 20, 8)                           # lies in several overlapping slices

    def det(crops):
        out = []
        for (x, y, w, h) in slice_grid(1024, 1024, 512, 0.2):
            gx, gy, gw, gh = ship
            inside = x <= gx and gx + gw <= x + w and y <= gy and gy + gh <= y + h
            out.append([Detection(gx - x, gy - y, gw, gh, 0.9)] if inside else [])
        return out

    r = sahi_detect(img, det, SahiConfig(512, 0.2))
    assert len(r.raw) > 1                              # seen by several slices ...
    assert len(r.detections) == 1                      # ... fused into one
    d = r.detections[0]
    assert (d.x, d.y, d.w, d.h) == ship


# ---------------- (7) and Table I
def test_record_roundtrip_and_size():
    r = SemanticRecord(0, 0.734, (1500, 33, 41, 9), 1.79e9, 7, P1, 34.74, 10.76)
    pkt = SemanticPacket(7, 1.79e9, [r, SemanticRecord(0, 0.3, (1, 2, 3, 4), 1.79e9, 7, P2)],
                         [(1, 1, b"\xff\xd8jpeg")])
    back = SemanticPacket.decode(pkt.encode())
    a, b = back.records
    assert a.bbox_global == r.bbox_global and a.image_id == 7 and a.timestamp == 1.79e9
    assert abs(a.confidence - 0.734) < 1 / 255 and a.lat == pytest.approx(34.74, abs=1e-4)
    assert b.lat is None and back.blobs == [(1, 1, b"\xff\xd8jpeg")]


def test_priority_levels_table_1():
    pol = PriorityPolicy(t_discard=0.25, t_context=0.4, t_meta=0.7)
    assert [pol.priority(p) for p in (0.1, 0.3, 0.5, 0.95)] == [P0, P3, P2, P1]
    assert PriorityPolicy(adaptive=False).priority(0.95) == P2
    assert PriorityPolicy(roi=False).priority(0.3) == P1


def test_packet_bytes_grow_with_priority():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    d = lambda p: [Detection(200, 200, 30, 10, p)]
    size = {p: len(build_packet(img, d(p), 1, 0.0).encode()) for p in (0.1, 0.3, 0.5, 0.9)}
    assert size[0.1] < size[0.9] < size[0.5] < size[0.3]   # P0 < P1 < P2 < P3


# ---------------- (8)-(17), (27)-(29)
def test_energy_equations():
    pw = PowerConfig(p_pre=1, p_tile=2, p_det=10, p_fusion=3, p_semantic=4, p_tx=8)
    e = processing_energy({"pre": 1, "tile": 1, "det": 2, "fusion": 1, "semantic": 0.5}, pw)
    assert e["total"] == pytest.approx(1 + 2 + 20 + 3 + 2)
    assert transmission_time(1e6, 8e6) == pytest.approx(1.0)
    assert comm_energy_direct(1e6, 8e6, 8) == pytest.approx(8.0)
    tot, e_isl, e_gs = comm_energy_relay(1e6, 80e6, 8e6, PowerConfig(p_isl=20, p_relay_gs=8))
    assert (e_isl, e_gs, tot) == pytest.approx((2.0, 8.0, 10.0))
    assert energy_saving(25, 100) == pytest.approx(0.75)
    assert downlink_reduction(1, 1000) == pytest.approx(0.999)
    assert total_latency(1, 2, 3, 4) == 10


def test_objective_respects_constraints():
    ref = {"E_total": 100.0, "D_tx": 100.0, "T_total": 100.0}
    rows = [{"E_total": 1, "D_tx": 1, "T_total": 1, "F1": 0.2, "IP": 1.0},     # cheapest, inaccurate
            {"E_total": 50, "D_tx": 50, "T_total": 50, "F1": 0.9, "IP": 0.95}]
    best = select_operating_point(rows, ref, ObjectiveConfig(a_min=0.5, i_min=0.9))
    assert best["E_total"] == 50 and best["J"] == pytest.approx(1.5)


# ---------------- links and path choice (16), (25), (26)
def test_link_schedule_spans_windows_and_queues():
    link = LinkSchedule([Window(0, 10, 8.0), Window(100, 200, 8.0)])   # 1 byte/s
    p = link.plan(5, 20)
    assert (p.start, p.finish, p.tx_s) == (5, 115, 20)
    link.commit(p)
    assert link.plan(0, 1).start == 115                                # FIFO behind it
    assert link.plan(0, 1e9) is None                                   # beyond the horizon


def test_relay_taken_when_direct_is_late_and_not_otherwise():
    pw = PowerConfig(p_tx=10, p_isl=20, p_relay_gs=10)
    direct = [Window(10_000, 10_600, 8e6)]
    isl = [Window(0, 600, 100e6)]
    relay_gs = [Window(1_000, 1_600, 8e6)]
    net = Network(direct, isl, relay_gs, pw, PathPolicy(deadline_s=5_000))
    ch = net.send(0, 1e6)
    assert ch.path == "relay" and ch.t_comm < 2_000 and set(ch.options) == {"direct", "relay"}
    net2 = Network(direct, isl, relay_gs, pw, PathPolicy(allow_relay=False, deadline_s=5_000))
    assert net2.send(0, 1e6).path == "direct"


# ---------------- metrics
def test_detection_metrics_and_ap():
    g = [Detection(0, 0, 10, 10, 1), Detection(50, 50, 40, 40, 1)]
    acc = DetAccumulator()
    acc.add([Detection(0, 0, 10, 10, 0.9), Detection(200, 200, 5, 5, 0.8)], g)
    m = acc.at(0.5)
    assert (m["TP"], m["FP"], m["recall"], m["precision"]) == (1, 1, 0.5, 0.5)
    assert acc.small_recall() == 1.0
    assert acc.ap() == pytest.approx(0.5)


def test_information_preservation_tolerance():
    ref = [Detection(0, 0, 10, 10, 0.9), Detection(100, 100, 10, 10, 0.9)]
    assert preserved(ref, [(0, 6, 5), (0, 300, 300)], tol_px=4) == [True, False]
    assert preserved(ref, [(1, 5, 5)], tol_px=4) == [False, False]          # wrong class


# ---------------- the whole onboard chain (Fig. 1) and the ground decoder
def test_onboard_pipeline_end_to_end():
    from sat7.pipeline import OnboardPipeline, PipelineConfig, ground_decode, record_centres
    from sat7.scenes import synthetic_scene
    from sat7.detectors import ClassicDetector
    img, gts = synthetic_scene(1024, 6, seed=3)
    pipe = OnboardPipeline(ClassicDetector(), PipelineConfig())
    out = pipe.run(img, image_id=5, timestamp=1.79e9)
    assert set(out.times) == {"pre", "tile", "det", "fusion", "semantic"}
    pkt, t_dec = ground_decode(out.packet)
    assert pkt.image_id == 5 and len(pkt.records) == len(out.detections) >= 5
    assert t_dec >= 0 and len(record_centres(pkt)) == len(pkt.records)
    # the same detections, re-used under another threshold, never grow in number
    hi = pipe.with_threshold(0.9)
    assert len(hi.detections(pipe.detect(img))[0]) <= len(out.detections)
