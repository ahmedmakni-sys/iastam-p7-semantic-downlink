"""Read-only access to the pipeline's outputs for the dashboard.

Nothing here computes a result: every number comes from a file a script wrote in results/.
REAL files have no suffix (paper_b0_b4.py --source airbus --detector yolo); SIM files end in
``_sim``. The dashboard never mixes the two. Runs launched from the dashboard's Simulation page
live in results/live/<id>/ with no suffix; their source key is ``live:<id>`` and their label
(SIM or REAL) is the one the script wrote, never guessed here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
SUFFIX = {"REAL": "", "SIM": "_sim"}
LIVE_PREFIX = "live:"

ABLATION_FR = {"B4 full": "B4 complet", "without SAHI": "sans SAHI",
               "without tile overlap": "sans chevauchement", "without fusion": "sans fusion",
               "without adaptive downlink": "sans politique adaptative",
               "without ROI transmission": "sans ROI", "without relay": "sans relais"}
PRIORITY_FR = {1: "P1 métadonnées", 2: "P2 + ROI", 3: "P3 + ROI + contexte"}
STRATEGY_FR = {"Bent pipe (raw, FIFO)": "Bent pipe (images brutes)",
               "Phi-sat-2 style, ocean only": "Phi-sat-2, océan seul",
               "Phi-sat-2 style, fair (+coastal)": "Phi-sat-2 équitable (+ côte)",
               "Ours: LoD + no buffer": "Nous : LoD sans mémoire",
               "Ours: LoD + FIFO": "Nous : LoD + FIFO",
               "Ours: LoD + value-greedy": "Nous : LoD + value-greedy"}


@dataclass
class PaperResults:
    """Outputs of scripts/paper_b0_b4.py for one source (REAL or SIM)."""
    label: str
    campaign: pd.DataFrame        # B0..B4
    ablations: pd.DataFrame
    sweep: pd.DataFrame           # B4 at several confidence thresholds
    packets: pd.DataFrame         # one row per B4 packet
    links: pd.DataFrame           # direct / isl / relay_gs windows
    scene: pd.DataFrame           # example scene: detections + ground truth
    scene_image: Path
    meta: dict
    joint: pd.DataFrame | None = None   # eq. (20) sweeps (weights, information floor)
    live_id: str | None = None          # set for a run launched from the Simulation page


def available_sources(results: Path = RESULTS) -> list[str]:
    """Committed sources with a complete set of files, REAL first."""
    return [s for s, suf in SUFFIX.items() if (results / f"paper_b0_b4{suf}.json").exists()]


def live_sources(results: Path = RESULTS) -> list[str]:
    """Complete live runs, most recent first, as source keys ``live:<id>``."""
    live = results / "live"
    if not live.is_dir():
        return []
    return [LIVE_PREFIX + d.name for d in sorted(live.iterdir(), reverse=True)
            if (d / "paper_b0_b4.json").exists()]


def source_files(source: str, results: Path = RESULTS) -> tuple[Path, str]:
    """(directory, suffix) holding a source's files."""
    if source.startswith(LIVE_PREFIX):
        return results / "live" / source[len(LIVE_PREFIX):], ""
    return results, SUFFIX[source]


def load_paper(source: str, results: Path = RESULTS) -> PaperResults:
    d, suf = source_files(source, results)

    def csv(name: str) -> pd.DataFrame:
        return pd.read_csv(d / f"{name}{suf}.csv")

    meta = json.loads((d / f"paper_b0_b4{suf}.json").read_text())
    joint = csv("paper_joint") if (d / f"paper_joint{suf}.csv").exists() else None
    live_id = source[len(LIVE_PREFIX):] if source.startswith(LIVE_PREFIX) else None
    return PaperResults(meta["label"], csv("paper_b0_b4"), csv("paper_ablations"),
                        csv("paper_tradeoff"), csv("paper_packets"), csv("paper_links"),
                        csv("paper_scene"), d / f"paper_scene{suf}.jpg", meta, joint, live_id)


COMPARE_METRICS = {"D_tx_MB": "Mo envoyés", "reduction_x": "réduction ×", "recall": "rappel",
                   "precision": "précision", "F1": "F1", "IP": "préservation (IP)",
                   "E_total_J": "énergie (J)", "ES": "énergie économisée",
                   "T_total_med_h": "latence méd. (h)", "relay_share": "part relais"}


def compare_campaigns(a: pd.DataFrame, b: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """B0..B4+J of two runs side by side: for each metric, its A, B and B - A columns."""
    a, b = a.set_index("config")[metrics], b.set_index("config")[metrics]
    configs = [c for c in a.index if c in b.index] + [c for c in b.index if c not in a.index]
    a, b = a.reindex(configs), b.reindex(configs)
    out = pd.DataFrame({f"{COMPARE_METRICS.get(k, k)} · {part}": col for k in metrics
                        for part, col in (("A", a[k]), ("B", b[k]), ("Δ", b[k] - a[k]))})
    out.index.name = "config"
    return out.reset_index()


def run_settings(meta: dict) -> dict:
    """The settings a run was made with, read back from its own paper_b0_b4.json."""
    s, j = meta.get("sahi", {}), meta.get("joint_eq20", {})
    w, c = j.get("weights", {}), j.get("constraints", {})
    return {"label": meta.get("label"), "source": meta.get("source"),
            "détecteur": meta.get("detector"), "scènes": meta.get("scenes"),
            "tuile (px)": s.get("tile"), "chevauchement": s.get("overlap"),
            "fusion": s.get("fusion"), "seuil conf.": s.get("conf_thr"), "liaison": meta.get("link"),
            "stations": ", ".join(meta.get("stations", [])), "échéance (h)": meta.get("deadline_h"),
            "α, β, γ": ", ".join(str(w.get(k)) for k in ("alpha", "beta", "gamma")),
            "A_min": c.get("a_min"), "I_min": c.get("i_min")}


@dataclass
class ImportedResults:
    """REAL measurements imported from oblhdj/iastam-p7-semantic-downlink (results/oblhdj)."""
    sahi: pd.DataFrame            # one row per (window, policy): recall, seam recall, compute
    swaths: dict                  # wp15_summary.json
    energy_stages: pd.DataFrame   # wp17 per-stage energy, REAL T_k x ASSUMED P_k
    relay: dict                   # wp20 latency direct vs relay


def load_imported(results: Path = RESULTS) -> ImportedResults | None:
    d = results / "oblhdj"
    if not (d / "wp16_swath_policies.json").exists():
        return None
    rows = []
    for f in ("wp16_swath_policies.json", "wp16_swath_policies_sahi512.json"):
        w = json.loads((d / f).read_text())
        for pol, v in w["policies"].items():
            if pol == "sahi" or f.endswith("policies.json"):     # whole/naive/edge do not depend
                rows.append({"policy": f"SAHI {w['sahi_window']} px" if pol == "sahi" else pol,
                             "recall": v["recall_overall"], "small": v["recall_small"],
                             "on_seam": v["recall_on_seam"],
                             "compute_x": v["compute_vs_regular_tiling"]})
    relay = json.loads((d / "wp20_relay_path_choice.json").read_text())
    return ImportedResults(pd.DataFrame(rows), json.loads((d / "wp15_summary.json").read_text()),
                           pd.read_csv(d / "wp17_energy_stages.csv"), relay)


@dataclass
class MeasuredResults:
    """The measured (REAL) results of the WP pipeline on Airbus tiles."""
    strategies: pd.DataFrame      # wp6_real_table.csv
    detector: dict                # wp1_metrics.json
    gate: dict                    # wp3_gate.json
    regret: pd.DataFrame          # wp8_queue_aware.csv, worst and mean regret per policy
    figures: dict                 # title -> png path, only those that exist


def load_measured(results: Path = RESULTS) -> MeasuredResults | None:
    if not (results / "wp6_real_table.csv").exists():
        return None
    wp8 = pd.read_csv(results / "wp8_queue_aware.csv")
    regret = (wp8.groupby("policy").regret.agg(pire="max", moyen="mean")
              .sort_values("pire").reset_index())
    figs = {"Navires livrés par stratégie": "wp6_real_recall.png",
            "Rappel du détecteur par taille": "wp1_recall_by_size.png",
            "Greedy contre optimum exact": "wp5_optimality_gap.png",
            "Effet de chaque hypothèse": "wp10_tornado.png",
            "Passages au-dessus de Sfax": "passes.png"}
    return MeasuredResults(pd.read_csv(results / "wp6_real_table.csv"),
                           json.loads((results / "wp1_metrics.json").read_text()),
                           json.loads((results / "wp3_gate.json").read_text()), regret,
                           {t: results / f for t, f in figs.items() if (results / f).exists()})


def ablation_deltas(ablations: pd.DataFrame) -> pd.DataFrame:
    """Each ablation against the full B4: relative change of recall, bytes, energy, latency."""
    full = ablations.set_index("config").loc["B4 full"]
    rows = []
    for _, r in ablations.iterrows():
        if r.config == "B4 full":
            continue
        rows.append({"ablation": ABLATION_FR.get(r.config, r.config),
                     "rappel": r.recall - full.recall,
                     "précision": r.precision - full.precision,
                     "octets (%)": r.D_tx_MB / full.D_tx_MB - 1,
                     "énergie (%)": r.E_total_J / full.E_total_J - 1,
                     "latence (%)": r.T_total_med_h / full.T_total_med_h - 1})
    return pd.DataFrame(rows)
