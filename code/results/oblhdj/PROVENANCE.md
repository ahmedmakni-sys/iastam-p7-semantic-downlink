# Imported results — oblhdj/iastam-p7-semantic-downlink

Copied unchanged from <https://github.com/oblhdj/iastam-p7-semantic-downlink> at commit
`d69f85d` (8 Oct 2026, MIT licence, same team). Both repositories share their history up to
`82914aa`; these files come from that repository's Phase 3 work (wp15–wp23) and are kept here as
measured reference points for the paper pipeline. Labels are the original repository's.

| file | what it is | label | produced by (in that repo) |
|---|---|---|---|
| `wp15_manifest.json`, `wp15_ships.csv`, `wp15_summary.json` | 24 swaths of 4 × 4 real Airbus test tiles (3072 px, 716 ships, 10.75 % on a seam). **Input** of `paper_b0_b4.py --source swaths` | REAL | `scripts/wp15_build_swaths.py` |
| `wp16_swath_policies.json/.csv` | YOLOv8n on those swaths: whole / naive grid / SAHI 768 px 20 % / edge-triggered | REAL | `scripts/wp16_swath_policies.py` |
| `wp16_swath_policies_sahi512.json/.csv` | the same with the paper's 512 px SAHI window | REAL | same, `--sahi-window 512` |
| `wp17_energy_model.json`, `wp17_energy_stages.csv` | per-stage energy with REAL stage times, ASSUMPTION powers | T_k REAL, P_k ASSUMPTION | `scripts/wp17_energy_model.py` |
| `wp18_campaign.json` | its B0–B4 validation run (B1/B2 on 200 tiles, B3 on the 40k-tile catalogue) | mixed, see file | `scripts/wp18_campaign_runner.py` |
| `wp19_relay_energy.json` | relay energy E_ISL + E_GS | D SIM, P ASSUMPTION | `scripts/wp19_relay_energy.py` |
| `wp20_isl_windows.csv`, `wp20_relay_path_choice.json` | ISL windows (RAAN +90°) and direct/relay choice | latency SIM, energy TARGET | `scripts/wp20_relay_path_choice.py` |
| `wp23_semantic_compare.json` | P0–P3 vs the LoD ladder through the WP scheduler | SIM over REAL detections | `scripts/wp23_semantic_compare.py` |

## Headline measurements (REAL)

SAHI on real swaths, YOLOv8n, conf 0.25 (wp16):

| policy | recall | small ships | ships on a seam | compute vs regular tiling |
|---|---|---|---|---|
| whole tiles (oracle) | 0.774 | 0.601 | 0.909 | 1.00× |
| naive regular grid | 0.728 | 0.561 | 0.675 | 1.56× |
| **SAHI 768 px, 20 %** | **0.772** | 0.612 | **0.922** | 1.56× |
| SAHI 512 px, 20 % (the paper's setting) | 0.771 | **0.637** | 0.896 | **4.00×** |

SAHI recovers the ships a naive grid cuts (+24.7 points on seam ships). At the paper's 512 px
it costs 4× the compute of regular tiling for the same overall recall (+2.5 points on small
ships only): a finding to state in the paper rather than hide.

## Known issue found while merging

`sat7/energy.py` (`e_comm_direct_J`, the relay adapters) and `sat7/joint.py` (`CostModel`) in
that repository divide **bytes** by a rate in **bits/s**, so their transmit time and E_comm are
8 times too small. The scripts that produced the files above do **not** have this error
(`wp17` uses bytes/s, `wp19` multiplies by 8). Only the "joint" section of
`wp23_semantic_compare.json` comes from the affected module. The ported `code/sat7/joint.py`
here uses `8 * bytes / R`.
