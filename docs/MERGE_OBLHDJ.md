# Merge with oblhdj/iastam-p7-semantic-downlink

[oblhdj/iastam-p7-semantic-downlink](https://github.com/oblhdj/iastam-p7-semantic-downlink) is a
teammate's fork of this project (shared history up to `82914aa`, MIT). Its Phase 3 work (wp15–wp23)
measured parts of the paper on real data that this repository had only simulated. This merge keeps
this repository's paper pipeline (`code/sat7/pipeline.py`, `scripts/paper_b0_b4.py`) as the base and
brings in what that fork does better. Source pinned at commit `d69f85d` (8 Oct 2026).

## What came in

| from the fork | here | what changed on the way |
|---|---|---|
| 24 real 3072 px swaths (wp15 manifest) | `sat7/scenes.py swaths_from_manifest`, `paper_b0_b4.py --source swaths`, `results/oblhdj/wp15_*` | the campaign rebuilds the **same** swaths, so B0–B4 here and SAHI there compare scene for scene |
| eq. (20) solved as one program (`sat7/joint.py`) | `code/sat7/joint.py`, config `B4+J`, `paper_joint.csv|.png` | **units fixed** (8 bits per byte); each level priced on this pipeline's **real** serialised bytes and **measured** ROI encoding time; confidence-weighted accuracy; an information-floor sweep, because the weight sweep is degenerate here |
| several ground stations (`orbit.find_passes_multi`, `max_gap_h`) | `code/sat7/orbit.py`, `relay.ground_windows(..., [stations])`, `--stations` | overlapping passes at two stations are **merged** (one radio, best rate), never counted twice; `max_gap_h` no longer overstates a gap after a nested pass |
| DOTA / YOLO / COCO (HRSID) loaders | `code/sat7/datasets.py`, `--source dota|coco` | boxes converted to `Detection`; loaders yield images ready for the campaign |
| REAL stage times for energy (`EnergyModel.from_results`) | `energy.detector_timing`, `PipelineConfig.det_ms`, `--det-timing yolo_gpu|yolo_cpu` | E_det = inferences × the measured YOLOv8n time, so a SIM run's energy is priced on the real detector |
| REAL measurements (wp15–wp23) | `code/results/oblhdj/` + `PROVENANCE.md`, dashboard page *Mesures réelles (fusion)* | copied unchanged, labels kept |

Tests: `code/tests/test_merge.py` (19 tests, partly adapted from the fork's own).

## Two findings from the merge

1. **Unit error in the fork's `sat7/energy.py` and `sat7/joint.py`**: they divide bytes by a rate in
   bits/s, so E_comm and transmit time are 8× too small. Its scripts `wp17` and `wp19` are correct,
   so its published ratios (8.9 : 1, ES ≈ 98 %) stand; only the joint section of `wp23` is affected.
   Worth reporting upstream.
2. **The paper's 512 px tile is expensive**: on the real swaths, SAHI at 512 px costs 4.00× the
   compute of regular tiling for the same overall recall (0.771 vs 0.772 at 768 px), +2.5 points on
   small ships only (wp16). The campaign can re-measure it: `--tile 768` vs the default 512.

## What still has to run on the GPU machine

Everything above runs here on SIM inputs. The REAL campaign needs `data/yolo_ships`, the WP1 weights
and `.venv312`:

```bash
cd code
.venv312/Scripts/python.exe scripts/paper_b0_b4.py --source swaths --detector yolo
.venv312/Scripts/python.exe scripts/paper_b0_b4.py --source swaths --detector yolo --tile 768 --tag _swaths768
.venv312/Scripts/python.exe scripts/paper_b0_b4.py --source swaths --detector yolo --stations sfax,svalbard --tag _swaths_2st
```

The first command writes the un-suffixed REAL files that the dashboard picks up automatically.
