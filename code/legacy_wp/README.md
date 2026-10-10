# legacy_wp — the archived WP pipeline

Before the paper's architecture was implemented, the project was built in work packages
(WP3 to WP14). That pipeline is archived here, **not deleted**, because it produced every
**measured** number on real Airbus tiles: 557× data reduction, 68.5 % of ships delivered,
greedy within 0.251 % of the exact optimum, and the sensitivity analysis. The narratives are in
[`reports/`](../../reports/).

Its onboard chain differs from the paper's:

| | WP pipeline (here) | paper pipeline (`code/sat7`, `code/scripts/paper_b0_b4.py`) |
|---|---|---|
| input | one 768 px Airbus tile | a large scene, 512 px tiles at 20 % overlap (SAHI) |
| onboard | learned gate → YOLO → level of detail (40 B / chip / ROI / tile) | SAHI + YOLO → fusion → records with priorities P0–P3 |
| downlink | value-greedy multi-pass scheduler | direct or inter-satellite relay, chosen by energy and latency |

## What is here

| path | content |
|---|---|
| `wplib/scheduler.py` | items, LoD encoders, FIFO / NoBuffer / ValueGreedy, `simulate` |
| `wplib/optimum.py` | exact DP optimum, fractional and whole-day bounds |
| `wplib/real_workload.py` | a day built from the real-detection catalogue |
| `scripts/` | wp3–wp14, `simulate_day`, `eval_prefilter`, `airbus_to_yolo`, and the PDF builders `build_report`, `build_overview`, `build_phase2_guide` |
| `tests/` | the WP tests: scheduler, optimum, real workload, queue-aware, fault injection, gate |

Shared with the paper pipeline and left in `code/`: `sat7/orbit|rle|dedup|prefilter|synthetic`,
the dataset build (`scripts/wp0_build_dataset.py`), the detector (`scripts/wp1_*`) and its
calibration (`scripts/wp2_*`). Every script still reads and writes `code/results/`, `code/data/`,
`code/runs/` and `code/models/`, exactly as before the move.

## Running it

From `code/`:

```bash
.venv/Scripts/python.exe -m pytest legacy_wp/tests -q
.venv/Scripts/python.exe legacy_wp/scripts/wp6_simulate_real.py      # the main measured result
.venv312/Scripts/python.exe legacy_wp/scripts/wp11_integration_demo.py --tiles 400
```

`rerun_all.sh` re-runs the whole analysis, WP and paper.
