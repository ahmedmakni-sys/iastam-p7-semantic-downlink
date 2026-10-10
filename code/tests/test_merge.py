"""What came in from oblhdj/iastam-p7-semantic-downlink (commit d69f85d): eq. (20) solved per object,
several ground stations, swaths from a wp15 manifest, DOTA / COCO loaders, REAL detector timing.
The solver and station tests are adapted from that repository's own tests."""
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from sat7.datasets import coco_ground_truth, iter_coco, iter_dota, parse_dota_lines, parse_yolo_lines
from sat7.energy import detector_timing
from sat7.joint import (INFO_CREDIT, P0, P1, P2, P3, CostModel, JointConstraints, JointWeights,
                        assemble_packet, encode_candidates, make_object, pareto, solve)
from sat7.orbit import (STATION_PRESETS, GroundStation, OrbitConfig, find_passes,
                        find_passes_multi, make_satellite, max_gap_h)
from sat7.pipeline import OnboardPipeline, PipelineConfig
from sat7.relay import Window, merge_windows
from sat7.sahi import Detection
from sat7.scenes import swaths_from_manifest
from sat7.semantic import SemanticPacket

RESULTS = Path(__file__).resolve().parents[1] / "results"
COST = CostModel()
SIZES = {P0: 0.0, P1: 24.0, P2: 940.0, P3: 3440.0}


def _obj(i, w=1.0, sizes=SIZES):
    return make_object(i, w, sizes, {}, COST)


# ---------------------------------------------------------------- eq. (20)
def test_units_bytes_are_eight_bits():
    """1 MB over 8 Mbit/s takes 1 s: the original module took 1/8 s."""
    c = CostModel(p_tx_W=10.0, r_tx_bps=8e6, p_proc_W=0.0)
    assert c.tx_time_s(1e6) == pytest.approx(1.0)
    assert c.energy_J(1e6, 0.0) == pytest.approx(10.0)


def test_unconstrained_optimum_is_the_lower_bound():
    r = solve([_obj(0), _obj(1)], JointWeights(0, 1, 0), JointConstraints(0.0, 0.0))
    assert all(lv == P0 for lv in r.assignment.values())
    assert r.J == r.J_lower_bound and r.gap == 0.0 and r.valid


def test_accuracy_floor_forces_reporting_at_the_cheapest_level():
    r = solve([_obj(0), _obj(1)], JointWeights(0, 1, 0), JointConstraints(1.0, 0.0))
    assert all(lv == P1 for lv in r.assignment.values()) and r.feasible and r.valid


def test_information_floor_climbs_levels_and_costs_more():
    objs = [_obj(0), _obj(1)]
    lo = solve(objs, JointWeights(0, 1, 0), JointConstraints(1.0, 0.0))
    hi = solve(objs, JointWeights(0, 1, 0), JointConstraints(1.0, 0.9))
    assert hi.info_preservation + 1e-9 >= 0.9 and hi.J >= lo.J and hi.valid


def test_constrained_J_never_below_the_bound():
    objs = [_obj(i, w) for i, w in enumerate((0.9, 0.5, 0.3, 0.8, 0.2))]
    for con in (JointConstraints(0.5, 0.3), JointConstraints(1.0, 0.8), JointConstraints(0.9, 0.9)):
        r = solve(objs, JointWeights(1, 1, 1), con)
        assert r.valid and r.feasible


def test_impossible_constraint_is_flagged_not_faked():
    assert not solve([_obj(0)], JointWeights(), JointConstraints(1.0, 1.01)).reachable
    assert not solve([_obj(0)], JointWeights(), JointConstraints(1.0001, 0.0)).reachable


def test_confidence_weights_report_the_likely_ships_first():
    """Accuracy is confidence-weighted: a floor below 100 % drops the least likely object."""
    objs = [_obj(0, 0.95), _obj(1, 0.9), _obj(2, 0.1)]
    r = solve(objs, JointWeights(0, 1, 0), JointConstraints(0.9, 0.0))
    assert r.assignment[2] == P0 and r.assignment[0] >= P1 and r.assignment[1] >= P1


def test_pareto_has_one_solution_per_weighting():
    rows = pareto([_obj(i) for i in range(3)], JointConstraints(1.0, 0.5))
    assert len(rows) >= 4 and all(r.valid for _, r in rows)


def test_candidates_are_measured_and_the_packet_matches_the_assignment():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    dets = [Detection(100, 100, 30, 10, 0.9), Detection(300, 300, 20, 20, 0.4)]
    objs, enc = encode_candidates(img, dets, COST)
    for o in objs:
        b = {op.level: op.bytes for op in o.options}
        assert b[P0] == 0 < b[P1] < b[P2] < b[P3]          # real bytes rise with the level
        assert {op.level: op.info for op in o.options}[P3] == pytest.approx(o.weight * INFO_CREDIT[P3])
    assign = {objs[0].id: P1, objs[1].id: P3}
    pkt = SemanticPacket.decode(assemble_packet([o.id for o in objs], assign, enc, 7, 1.0).encode())
    assert [r.priority for r in pkt.records] == [P1, P3]
    assert sorted(k for _, k, _ in pkt.blobs) == [1, 2]    # only the P3 object sends ROI + context


# ---------------------------------------------------------------- ground stations
START = datetime(2026, 9, 18, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def sat():
    return make_satellite(OrbitConfig(epoch=START))


def test_passes_are_tagged_with_their_station(sat):
    passes = find_passes(sat, GroundStation(), START, 12)
    assert passes and all(p.station == GroundStation().name for p in passes)


def test_a_station_network_adds_contacts_and_shortens_the_gap(sat):
    sfax = STATION_PRESETS["sfax"]
    single = find_passes(sat, sfax, START, 12)
    multi = find_passes_multi(sat, [sfax, STATION_PRESETS["svalbard"]], START, 12)
    assert len(multi) > len(single) and multi == sorted(multi, key=lambda p: p.rise)
    assert max_gap_h(multi, START, 12) <= max_gap_h(single, START, 12) + 1e-9
    assert max_gap_h([], START, 12) == 12


def test_overlapping_windows_never_count_capacity_twice():
    merged = merge_windows([Window(0, 100, 8.0), Window(50, 150, 16.0)])
    assert [(w.start, w.end, w.rate_bps) for w in merged] == [(0, 50, 8.0), (50, 150, 16.0)]
    assert sum(w.capacity_bytes for w in merged) == pytest.approx(50 + 200)


# ---------------------------------------------------------------- scenes and datasets
def test_swaths_are_rebuilt_from_a_manifest(tmp_path):
    img_dir, lbl_dir = tmp_path / "images" / "test", tmp_path / "labels" / "test"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    names = []
    for k in range(4):
        name = f"t{k}.jpg"
        cv2.imwrite(str(img_dir / name), np.full((768, 768, 3), 40 * k, np.uint8))
        (lbl_dir / f"t{k}.txt").write_text("0 0.5 0.5 0.1 0.05\n")
        names.append(name)
    man = {"split": "test", "tile_px": 768, "stride_px": 768,
           "swaths": [{"swath_id": "swath_000",
                       "source_tiles": [[k // 2, k % 2, n] for k, n in enumerate(names)]}]}
    (tmp_path / "m.json").write_text(json.dumps(man))
    (sid, img, gts), = list(swaths_from_manifest(tmp_path / "m.json", tmp_path))
    assert sid == "swath_000" and img.shape == (1536, 1536, 3) and len(gts) == 4
    g = sorted(gts, key=lambda d: (d.y, d.x))[3]          # the tile at row 1, col 1
    assert (g.x, g.y) == pytest.approx((768 + 345.6, 768 + 364.8))


def test_the_imported_manifest_describes_24_real_swaths():
    man = json.loads((RESULTS / "oblhdj" / "wp15_manifest.json").read_text())
    assert len(man["swaths"]) == 24 and man["swath_px"] == 3072
    assert all(len(s["source_tiles"]) == 16 for s in man["swaths"])


def test_dota_lines_become_top_left_boxes():
    lines = ["imagesource:GE", "gsd:0.3", "100 100 140 100 140 160 100 160 ship 0",
             "0 0 10 0 10 10 0 10 harbor 0"]
    (b,) = parse_dota_lines(lines)
    assert (b.x, b.y, b.w, b.h) == (100.0, 100.0, 40.0, 60.0)
    assert len(parse_dota_lines(lines, classes=None)) == 2


def test_yolo_and_coco_ground_truth():
    (b,) = parse_yolo_lines(["0 0.5 0.5 0.1 0.2", "3 0.1 0.1 0.05 0.05"], 1000, 1000, classes={0})
    assert (b.x, b.y, b.w, b.h) == (450.0, 400.0, 100.0, 200.0)
    coco = {"categories": [{"id": 1, "name": "ship"}, {"id": 2, "name": "plane"}],
            "images": [{"id": 7, "file_name": "a.png", "width": 800, "height": 600}],
            "annotations": [{"image_id": 7, "category_id": 1, "bbox": [10, 20, 40, 60]},
                            {"image_id": 7, "category_id": 2, "bbox": [0, 0, 5, 5]}]}
    (g,) = coco_ground_truth(coco)["a.png"]
    assert (g.x, g.y, g.w, g.h) == (10, 20, 40, 60)


def test_dataset_iterators_pair_images_and_labels(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "ann").mkdir()
    cv2.imwrite(str(tmp_path / "img" / "P1.png"), np.zeros((64, 64, 3), np.uint8))
    (tmp_path / "ann" / "P1.txt").write_text("1 1 9 1 9 9 1 9 ship 0\n")
    (name, img, gts), = list(iter_dota(tmp_path / "img", tmp_path / "ann"))
    assert name == "P1.png" and img.shape == (64, 64, 3) and len(gts) == 1
    coco = {"categories": [{"id": 1, "name": "ship"}],
            "images": [{"id": 1, "file_name": "P1.png", "width": 64, "height": 64}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 8, 8]}]}
    (tmp_path / "c.json").write_text(json.dumps(coco))
    assert len(list(iter_coco(tmp_path / "c.json", tmp_path / "img"))) == 1


# ---------------------------------------------------------------- energy on a REAL detector time
def test_detector_timing_reads_the_wp1_measurements():
    if not (RESULTS / "wp1_metrics.json").exists():
        pytest.skip("wp1 results absent")
    gpu, cpu = detector_timing(RESULTS, "yolo_gpu"), detector_timing(RESULTS, "yolo_cpu")
    assert 0 < gpu.ms_per_inference < cpu.ms_per_inference and "REAL" in gpu.source


def test_det_ms_prices_every_inference():
    img = np.zeros((1024, 1024, 3), np.uint8)
    det = lambda crops: [[] for _ in crops]               # noqa: E731
    out = OnboardPipeline(det, PipelineConfig(det_ms=10.0)).run(img, 0, 0.0)
    assert out.times["det"] == pytest.approx(9 * 0.010)    # 3 x 3 slices of 512 px at 20 %
