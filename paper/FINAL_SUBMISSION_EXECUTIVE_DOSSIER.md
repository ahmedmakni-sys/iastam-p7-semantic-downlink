# 🛰️ Final Submission & Executive Dossier
# "Transmitting Information Rather Than Raw Data"
## IASTAM 6.0 — Track 4, Problem 7 (IEEE IAS Tunisia Congress)

---

## 1. Executive Summary & Problem Framing

Earth Observation (EO) satellites in Low Earth Orbit (LEO) produce data at rates orders of magnitude beyond what downlink channels can deliver. A standard high-resolution sensor easily captures **60+ GB of raw imagery per day**, while ground contact time with a regional ground station—such as our modeled ground station at **Sfax, Tunisia** ($34.7^\circ\text{N}, 10.7^\circ\text{E}$)—is restricted to **~34 minutes across 5 discrete passes daily**.

Traditional satellite architectures rely on a *"Bent-Pipe"* approach: store raw imagery and blast it down over ground stations, relying on aggressive spatial compression (JPEG2000). This approach fails because **80% to 95% of ocean tiles are empty water or opaque cloud**.

Our solution shifts the paradigm: **stop downlinking pictures; downlink what the satellite learned**.

```
                   ONBOARD ENGINE                                              GROUND
 ┌─────────┐
 │ Imager  │ 768x768 tiles
 └────┬────┘
      ▼
 ┌──────────────────────┐  Empty sea (40.2% skipped)
 │ [1] Learned Gate     │ ──────────────────────────► [Discarded / 0 Bytes]
 │     47k-param CNN    │  0.052 ms GPU, 98.8% safety
 └────┬─────────────────┘
      │ Suspect tiles
      ▼
 ┌──────────────────────┐  mAP50: 0.804, P: 0.817, R: 0.729
 │ [2] YOLOv8 Detector  │  10.7 ms GPU / 38.6 ms CPU (ONNX FP32)
 └────┬─────────────────┘
      │ Detections & Confidence
      ▼
 ┌──────────────────────┐  Isotonic calibration (ECE: 0.0878 -> 0.0099)
 │ [3] LoD Encoder      │  • Confident (≥0.670): 40 Bytes metadata
 │     Confidence-Aware │  • Ambiguous (0.25-0.67): ~900 Bytes crop
 │                      │  • Wake / Dark vessel: ~2.5 kB ROI
 └────┬─────────────────┘  • Coastal tile: adaptive q40 JPEG
      │ Items with (value, size)
      ▼
 ┌──────────────────────┐  Buffer feedback loop (WP8)
 │ [4] Value-Scheduler  │ ◄────────────────────────┐
 │     Multi-Pass       │  Knapsack refills each pass over Sfax
 └────┬─────────────────┘  0.251% gap from exact dynamic program
      │ Space Packets (CCSDS 133.0-B)
      ▼
 ┌──────────────────────┐                             ┌──────────────────────────────┐
 │ [5] Downlink Passes  │ ═══════════════════════════►│ Ground Station (Sfax)        │
 │     5 passes / 36 h  │                             │ 100% onboard knowledge kept  │
 └──────────────────────┘                             └──────────────────────────────┘
```

---

## 2. Key Verified Findings (Measured on 53,195 Real Airbus Tiles)

Every number below is strictly empirically verified from our dataset and automated test suite:

| Metric | Measured Value | Comparison / Engineering Context |
|---|---|---|
| **Data Reduction vs Bent Pipe** | **557×** | 60,124 MB offered $\rightarrow$ 108.0 MB sent per day. |
| **Delivery under Congestion** | **2.26× vs FIFO** | At 160k tiles/day load, FIFO drops high-value targets; our value-greedy scheduler delivers 2.26× more ships. |
| **Onboard Knowledge Delivered** | **100.0%** | Under nominal orbital pass capacity, the downlink is *not* the bottleneck; 100% of detected targets are delivered. |
| **Scheduler Optimality Gap** | **0.251%** | Measured against an exact Dynamic Programming knapsack solver (`legacy_wp/wplib/optimum.py`), verified vs brute force. |
| **Pre-Filter Gate Speed & Safety** | **0.052 ms / 98.8%** | 47k-parameter CNN drops 40.2% of empty tiles while keeping 98.8% of ships (outperforms classic CV by +43.3 points). |
| **Detector Precision & Recall** | **mAP50 0.804** | Recall by size: small (<32px) 0.739, medium 0.958, large 0.994 (at conf $\ge$ 0.05). |
| **Edge Compute Selection** | **ONNX FP32** | Dynamic INT8 is 8.5× slower on CPU and loses 2.2 points on small ships. FP32 is lossless and 1.21× faster than PyTorch. |
| **Calibration Gain** | **ECE cuts 8.9×** | Detector is naturally underconfident (0.352 vs 0.408); isotonic regression sets true high-confidence threshold at **0.670** (saving 8.1% bytes). |

---

## 3. Merging the Core Ideas: Spaceborne Engineering Realities

A software pipeline that works on a desktop GPU will fail in space without addressing physical constraints. Our architecture explicitly mitigates the 5 major spaceborne constraints documented in the literature (*Duggan et al., 2025, European Journal of Remote Sensing*):

### A. Thermal & Duty-Cycle Constraint (Vacuum Physics)
* **The Constraint:** Space has no air. Heat cannot dissipate via convection. Running a GPU continuously at 100% causes thermal throttling and hardware shutdown within minutes.
* **Our Solution:** The pipeline operates under an orbital **Duty Cycle (ConOps)**. Orbit propagation over Sfax demonstrates that target maritime areas require only **3 to 5 minutes of active compute per pass**. The 47k-parameter gate (0.052 ms) quickly eliminates empty ocean, keeping the high-power detector asleep for 40% of the swath.

### B. SWaP-C & Compute Trade-Off (Why We Tested and Rejected Dynamic INT8)
* **The Reality:** While literature often recommends INT8 blindly, our empirical benchmarks (**Report 02**) revealed that dynamic INT8 quantization on CPU is **8.5× slower** (328.8 ms vs 38.6 ms) and causes an unacceptable **2.21 point drop on small ships**.
* **Our Decision:** We adopted **ONNX FP32**, which delivers bit-identical recall to PyTorch and runs 1.21× faster on embedded CPU cores.

### C. Seam Crossings & Tile Boundaries (The SAHI Dilemma)
* **The Reality:** In our real 10k swath analysis (**Report 13**), large ships (380 px) straddle tile seams **74.5% of the time**.
* **Our Finding:** Blanket SAHI (sliding window with 20% overlap) imposes a **1.51× compute penalty** across the entire swath. However, our seam analysis proved that partially cut ships are still detected on at least one tile. We therefore recommend an edge-triggered selective overlap rather than wasteful global slicing.

### D. Downlink Telemetry vs Radio Bit-Flips (CCSDS Compliance)
* **The Problem:** Standard compression archives (`.tar.gz`) are catastrophic in space: a single cosmic-ray bit-flip corrupts the entire archive.
* **Our Solution:** Telemetry is structured as **CCSDS 133.0-B Space Packets** using **Protocol Buffers**. Every detection is an independent, checksummed packet with forward error correction. If packet #14 suffers fading, packet #15 is decoded without error.

### E. Maritime Mission Intelligence: Dark Vessels & Wake Dynamics
* **Tactical Context:** Operators on the ground need more than a tight bounding box:
  * **Wake Extension (L2 Rung):** Allocating 2.5 kB to capture the vessel's Kelvin wake allows ground analysts to calculate speed and nautical heading.
  * **AIS Correlation:** Vessels transmitting active Automatic Identification System (AIS) positions receive 40-byte metadata tokens. Vessels detected with **no AIS signal** ("Dark Vessels" engaged in illicit trafficking or unregulated fishing) are automatically escalated to Priority 1 (full image chip + wake).

---

## 4. Product Level-of-Detail (LoD) Ladder

Our byte budget (**Report 04 & 06**) is structured into a dynamic value ladder:

```
[Level 0] 40 Bytes  ──► Confident detection (conf ≥ 0.670): Lat/Lon, Class, BBox, AIS ID.
[Level 1] ~900 B    ──► Ambiguous detection: JPEG2000 image chip centered on hull.
[Level 2] ~2.5 kB   ──► High-priority / Dark vessel / Wake: extended crop for speed & heading.
[Coastal] ~42.9 kB  ──► Coastal scene: adaptive queue-aware JPEG (q40) under low buffer pressure.
[Thumbl.] ~997 B    ──► Low-resolution context thumbnail (gated by detector/gate agreement).
```

---

## 5. Defense & Jury Q&A Strategy (IEEE IAS Tunisia)

When presenting to academic and industrial judges, use these data-backed defenses:

#### Q1: "Why is your headline ship delivery 0.685 instead of 95%+?"
* **Answer:** *"Most papers report synthetic recalls on clear skies. In our work, we refuse to make up numbers. On real Airbus imagery with an assumed 15% regional cloud cover, 0.685 represents **100% of all ships the satellite still knows about**. The loss is dominated by optical cloud cover, not our downlink. Across a 0% to 50% cloud sweep, our delivery spans 0.40 to 0.82. We provide the full honest band rather than a single inflated claim."*

#### Q2: "Why not just use traditional JPEG2000 compression on all tiles?"
* **Answer:** *"In our byte budget audit (Report 06), transmitting full tiles—even at quality 40—consumes 62% of the downlink budget. Compressing harder hurts small ship recall by 4.0 points. Our semantic approach delivers a **557-fold data reduction** because confident detections require only 40 bytes. We transmit actionable intelligence, not ocean water."*

#### Q3: "Did you actually prove your scheduler is optimal?"
* **Answer:** *"Yes. In Report 08, we formulated the multi-pass pass-allocation problem as a bounded knapsack problem and developed an exact Dynamic Programming solver (`legacy_wp/wplib/optimum.py`), verified against brute force. Our value-greedy scheduler achieves an optimality gap of only **0.251%** at operational scale, while running in sub-millisecond execution time."*

#### Q4: "How does your system perform when the satellite experiences traffic bursts?"
* **Answer:** *"Under heavy congestion (160,000 tiles/day), traditional FIFO queues drop high-value detections blindly. Our value-aware queue scheduler achieves **2.26× higher vessel delivery** than FIFO. Furthermore, our queue-aware encoder (Report 07) senses buffer pressure and dynamically throttles coastal tiles to protect critical vessel telemetry."*

---

## 6. Deliverable Checklist for Submission (26 Sept)

- [x] **Technical Codebase:** 15 Work Packages complete, 86 tests passing (`pytest tests`).
- [x] **Reproducible Results:** 15 narrative markdown reports in `reports/` backed by authoritative CSVs in `code/results/`.
- [x] **Technical Report (PDF):** `paper/IASTAM_P7_Technical_Report.pdf` (compiled via `python legacy_wp/scripts/build_report.py`).
- [x] **Project Overview (PDF):** `paper/IASTAM_P7_Project_Overview.pdf`.
- [x] **Poster Guide (PDF):** `paper/IASTAM_P7_Phase2_Poster_Guide.pdf`.
- [x] **2-Minute Video Script:** `paper/VIDEO_SCRIPT_2MIN.md` (timed second-by-second with visuals).
- [x] **Comprehensive Dossier:** This document (`paper/FINAL_SUBMISSION_EXECUTIVE_DOSSIER.md`).
