# Paper → code

*Energy-Aware Semantic Downlink for Maritime Object Detection with Optional Inter-Satellite
Relay* (IASTAM, ENIS). Where every section and equation of the paper lives in `code/`.

| Paper | What it specifies | Code |
|---|---|---|
| Fig. 1–2, eq. 9 | the whole onboard chain, stage by stage, and the ground decoder | `sat7/pipeline.py` `OnboardPipeline`, `PipelineConfig`, `ground_decode` |
| III-B, eq. 1 | the large image | `sat7/scenes.py` (24 real swaths, mosaics, synthetic), `sat7/datasets.py` (DOTA, HRSID/COCO) |
| III-B, eq. 1–2 | large image, overlapping tiles, 512 × 512 at ~20 % overlap, configurable | `sat7/sahi.py` `slice_grid`, `SahiConfig(tile, overlap)` |
| III-C, eq. 3–4 | SAHI: the detector on every tile, `d = (x, y, w, h, p, c)` | `sat7/sahi.py` `sahi_detect`, `Detection`; `sat7/detectors.py` `YoloDetector` |
| III-D, eq. 5 | `x_g = x_0 + x_l`, `y_g = y_0 + y_l` | `Detection.to_global` |
| III-D, eq. 6 | fusion, suppression or merging, thresholds are parameters | `sat7/sahi.py` `fuse(mode="nms"\|"nmm", metric="iou"\|"ios", thr)` |
| IV-A, eq. 7 | record `{class, confidence, bbox_global, timestamp, image_id}` (+ geo) | `sat7/semantic.py` `SemanticRecord`, binary `SemanticPacket.encode/decode` |
| IV-B, Table I | priorities P0–P3, thresholds calibrated experimentally | `PriorityPolicy(t_discard, t_context, t_meta)`, `build_packet`, ROI + context JPEGs |
| V-A..C, eq. 8–14 | `E_total = E_proc + E_comm`, `E_k = P_k T_k`, `E_comm = P_tx D_tx / R_tx` | `sat7/energy.py` |
| V-D, eq. 15–16 | relay energy `E_ISL + E_GS`, `min(E_direct, E_relay)` within windows and deadline | `sat7/energy.py` `comm_energy_relay`; `sat7/relay.py` `Network.send` |
| V-E, eq. 17–19 | `ES = 1 − E_proposed / E_baseline` | `energy_saving` |
| VI, eq. 20–22 | `min αE + βD + γT` s.t. accuracy ≥ A_min, information preservation ≥ I_min | `sat7/joint.py` `solve` (level P0–P3 chosen per object, lower bound + repair), `pareto`, `constraint_sweep`; campaign config `B4+J`. Also `energy.select_operating_point` (best confidence threshold) |
| VII, eq. 23–26 | direct path, relay path, `J = λ_E E + λ_T T` | `sat7/relay.py` `PathPolicy`, `Network` (SGP4 passes + ISL geometry); one station or a network (`orbit.find_passes_multi`, `--stations`) |
| VIII-B, eq. 31 | B0 → B1 → B2 → B3 → B4 | `scripts/paper_b0_b4.py` `CONFIGS` |
| VIII-D | testbed: detector, SAHI, packet encoder, comm simulator; all parameters configurable | the modules above; CLI flags of `paper_b0_b4.py`; `PowerConfig`, `RelayConfig`, `LinkConfig` |
| VIII-E, eq. 27–30 | DR, precision/recall/F1/mAP, small-object recall, information preservation, latency, energy | `sat7/metrics.py`, `sat7/energy.py` |
| VIII-G | six ablations | `paper_b0_b4.py` `ABLATIONS` |
| VIII-I, eq. 32–35 | trade-off curves | `results/paper_tradeoffs*.png` |

Every equation is pinned by a test in `code/tests/test_paper.py`.

Merged from oblhdj/iastam-p7-semantic-downlink: `joint`, `datasets`, the swaths, the station network and the REAL SAHI measurements — see [`MERGE_OBLHDJ.md`](MERGE_OBLHDJ.md).

## Running it

```bash
cd code
# anywhere, no dataset: synthetic 2048 px scenes + the classic fallback detector -> labelled SIM,
# with E_det priced on the REAL YOLOv8n CPU time
.venv/Scripts/python.exe scripts/paper_b0_b4.py --det-timing yolo_cpu
# the real campaign: the 24 real 3072 px swaths (same as oblhdj wp16) + the trained YOLO
.venv312/Scripts/python.exe scripts/paper_b0_b4.py --source swaths --detector yolo
# the paper's recommended large-scene sets
.venv312/Scripts/python.exe scripts/paper_b0_b4.py --source dota --dota-images D/images --dota-labels D/labelTxt --detector yolo
.venv312/Scripts/python.exe scripts/paper_b0_b4.py --source coco --coco-json H/annotations/test2017.json --coco-images H/JPEGImages --detector yolo
```

## What is measured and what is assumed

- **Measured:** detections; the size of every packet (real binary records and real JPEG ROIs);
  every stage time `T_k` on the machine that runs the script; ground passes and ISL geometry (SGP4).
- **Assumed** (and printed on every run): every power `P_k` (`PowerConfig`), link and ISL rates
  (`LinkConfig`, `RelayConfig`), the relay's orbit, capture times, the deadline, the eq. 20 weights and floors, and the eq. 25/26 weights.
- Stage times come from a laptop, not flight hardware, so energy figures are **estimates**.
- **Datasets:** the paper suggests xView, VisDrone, DOTA and HRSID. The implementation uses
  **Airbus Ship Detection** (optical, the only dataset prepared in this repository). Real
  scenes are the 24 swaths of 4 × 4 test tiles (or 2 × 2 mosaics), so the seams between stitched
  tiles are artificial. Loaders for DOTA and HRSID exist (`sat7/datasets.py`); the detector would
  need fine-tuning on them first.
- **SIM results** (`results/*_sim.*`) come from synthetic scenes and the classic fallback
  detector. They show that the chain works; they are not YOLO results and must not be quoted as such.
