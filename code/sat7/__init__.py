"""sat7 -- IASTAM 6.0, Track 4, Problem 7: the paper's semantic downlink pipeline.

"Energy-Aware Semantic Downlink for Maritime Object Detection with Optional Inter-Satellite
Relay" (ENIS). Each module implements one part of the paper:

pipeline   the onboard chain, stage by stage (Fig. 1-2, eq. 9) + ground decoder
sahi       512 px / 20% overlap tiling, SAHI, global coordinates, fusion   (eqs. 1-6)
detectors  YOLO adapter + classic fallback detector
semantic   semantic records, P0-P3 priorities, ROI/context, packet codec   (eq. 7, Table I)
energy     processing/communication energy, latency, DR, REAL detector timing (eqs. 8-19, 27-30)
joint      the joint objective solved per object, P0-P3 per detection      (eqs. 20-22)
relay      ground passes (one station or a network), ISL windows, path choice (eqs. 15-16, 23-26)
metrics    precision/recall/F1/mAP, small-object recall, info preservation (VIII-E)
scenes     large scenes: the 24 real swaths, Airbus mosaics, synthetic      (eq. 1)
datasets   the paper's recommended large-scene sets: DOTA, HRSID / COCO     (VIII-C)

joint, datasets, the swaths and the station network are merged from
oblhdj/iastam-p7-semantic-downlink (docs/MERGE_OBLHDJ.md).

Foundation shared with the archived WP pipeline (code/legacy_wp/):
orbit      SGP4 orbit, ground stations, per-pass downlink capacity
rle        Airbus Ship Detection run-length encoding <-> masks <-> boxes
dedup      near-duplicate tiles, leakage-free train/val/test split
prefilter  classic computer-vision pre-filter (Otsu, morphology, watershed)
synthetic  synthetic sea tiles with known ships (tests and SIM runs only)
"""
