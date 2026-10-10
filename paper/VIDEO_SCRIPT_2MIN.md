# 🎬 2-Minute Video Script: "Transmitting Information Rather Than Raw Data"
## IASTAM 6.0 — Track 4, Problem 7 (IEEE IAS Tunisia Congress)

* **Duration:** Exactly 2 minutes (120 seconds).
* **Word Count:** ~260 words (comfortable speaking pace: ~130 words/minute).
* **Language:** English (standard IEEE competition language, with French notes).
* **Visual Assets Used:** `docs/simple_steps.png`, `docs/architecture.png`, `code/results/wp11_funnel.png`, `code/results/passes.png`, and demo screens.

---

### Timing & Storyboard Table

| Timestamp | Visual on Screen | Voiceover (Narrator Speech) | Key Message / Data Shown |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:15**<br>(15s) | **Slide 1:** Earth Observation satellite over the Mediterranean / Sfax ground station.<br>Graphic of a massive pipe shrinking into a tiny needle.<br>*Text: 60 GB captured vs 108 MB contact window.* | *"Earth observation satellites capture vastly more ocean than their downlink can ever deliver. In our real orbit simulation over Sfax, a small satellite collects over 60 gigabytes daily, but gets only 34 minutes of radio contact. Traditional compression can't bridge this gap. Our answer? Stop sending raw pixels. Send what the satellite learned."* | **The Bottleneck:**<br>Downlink is limited by physics, contact windows, and RF budget. |
| **0:15 – 0:35**<br>(20s) | **Slide 2:** The 2-stage Onboard Engine.<br>Animation showing 47k-param Gate $\rightarrow$ YOLOv8 768px.<br>*Highlight: 0.052 ms gate, 11 ms detector.* | *"Our onboard chain uses a two-tiered architecture. First, a lightweight 47,000-parameter neural gate drops 40% of empty ocean tiles in just 50 microseconds, safely preserving 98.8% of ships. Only candidate scenes reach our fine YOLOv8 detector, operating in lossless FP32 at 11 milliseconds per tile."* | **Compute Efficiency:**<br>Learned gate eliminates waste before heavy inference. |
| **0:35 – 0:55**<br>(20s) | **Slide 3:** Confidence-Aware Level-of-Detail (LoD).<br>Visual ladder: 40 B metadata $\rightarrow$ 900 B crop $\rightarrow$ 2.5 kB wake/context $\rightarrow$ coastal tile. | *"Next, we encode detections into a Confidence-Aware Level of Detail. Highly certain ships cost just 40 bytes of telemetry. Ambiguous detections earn a 900-byte crop. Vessels with wakes are allocated 2.5 kilobytes to estimate speed and heading. And uncertain coastal tiles are transmitted adaptively."* | **Semantic Downlink:**<br>Bytes are spent in proportion to uncertainty & tactical value. |
| **0:55 – 1:20**<br>(25s) | **Slide 4:** Value-Aware Multi-Pass Scheduler.<br>Plot of knapsack queue, orbit pass timeline over Sfax, buffer pressure feedback. | *"Because downlink passes are discrete knapsacks, our value-aware scheduler fills each contact window greedily by value-per-byte. When the buffer experiences congestion, queue-aware feedback dynamically throttles coastal imagery to protect critical vessel alerts. Exact dynamic programming proves our scheduler is within 0.25% of the theoretical optimum."* | **Mathematical Rigor:**<br>Scheduler is 0.25% from exact optimum (no hand-waving). |
| **1:20 – 1:45**<br>(25s) | **Slide 5:** Real Empirical Results.<br>Bar charts from reports: 557× data reduction, 2.26× over FIFO, mAP50 0.804.<br>*Tag: [REAL] on 53,195 Airbus tiles.* | *"We tested this on 53,195 real Airbus satellite tiles. The result? A 557-fold data reduction—slashing 60 gigabytes offered down to 108 megabytes sent. Under heavy orbital congestion, our system delivers 2.26 times more vessels than standard FIFO queues, delivering 100% of all information discovered onboard."* | **Empirical Proof:**<br>557× bandwidth reduction, 2.26× vs FIFO, 100% onboard knowledge delivered. |
| **1:45 – 2:00**<br>(15s) | **Slide 6:** Flight Readiness & Conclusion.<br>CCSDS space packets, space hardware thermal duty-cycle, logo of IASTAM 6.0.<br>*Text: Validated on real telemetry. Ready for orbit.* | *"Designed for spaceborne constraints—with thermal duty-cycling, fault-tolerant CCSDS space packets, and AIS dark-vessel correlation—this pipeline turns satellites from blind data recorders into autonomous decision engines. Thank you."* | **Closing Impact:**<br>Robust, fault-tolerant, space-ready. |

---

### Production Notes for Recording

1. **Tone:** Crisp, confident, academic yet punchy. Avoid vague adjectives; quote the verified numbers (**557×**, **0.25% gap**, **40 bytes**, **2.26× vs FIFO**).
2. **Audio Track:** Clean vocal recording over subtle, low-volume technological ambient background music (optional).
3. **Screen Captures to Include:**
   * `code/results/wp11_funnel.png` (shows the data reduction funnel from 400 tiles down to telemetry).
   * `docs/simple_steps.png` (clean block diagram).
   * `docs/architecture.png` (the cross-check and buffer pressure loops).
