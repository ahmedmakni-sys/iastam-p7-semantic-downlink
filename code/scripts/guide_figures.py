"""Figures for the French pipeline guide: pipeline flow, file relations, dashboard mock-ups.

The mock-ups are drawn from the real campaign outputs (results/paper_*_sim.csv) and from the
real link geometry (sat7.relay), so every number on them is one the pipeline produced.

    .venv/Scripts/python.exe scripts/guide_figures.py
    -> results/guide/*.png   (then: python scripts/build_pipeline_guide.py)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from sat7.detectors import ClassicDetector
from sat7.orbit import LINK_PRESETS, GroundStation, OrbitConfig
from sat7.pipeline import OnboardPipeline, PipelineConfig
from sat7.relay import RelayConfig, ground_windows, isl_windows, relay_orbit
from sat7.sahi import slice_grid
from sat7.scenes import synthetic_scene

RES = ROOT / "results"
OUT = RES / "guide"
NAVY, OCEAN, ORANGE = "#0B2545", "#1B6CA8", "#E07A1F"
GREEN, RED, PURPLE, GREY = "#2E8B57", "#C0392B", "#7D3C98", "#6B7280"
LIGHT, CARD = "#F3F6FA", "#FFFFFF"
PRIO = {1: GREEN, 2: ORANGE, 3: RED}
FR_ABL = {"without SAHI": "sans SAHI", "without tile overlap": "sans chevauchement",
          "without fusion": "sans fusion", "without adaptive downlink": "sans politique adaptative",
          "without ROI transmission": "sans ROI", "without relay": "sans relais"}


def box(ax, x, y, w, h, text, fc=CARD, ec=OCEAN, fs=8.5, tc=NAVY, bold=False, lw=1.3, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, a, b, color=GREY, lw=1.2, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- 1. pipeline flow
def pipeline_flow():
    fig, ax = canvas(12, 5.6)
    ax.add_patch(Rectangle((0.01, 0.50), 0.98, 0.44, fc="#EAF2FA", ec="none"))
    ax.add_patch(Rectangle((0.01, 0.05), 0.98, 0.40, fc="#FDF1E6", ec="none"))
    ax.text(0.02, 0.91, "À BORD DU SATELLITE PRINCIPAL  —  sat7/pipeline.py (OnboardPipeline.run)",
            fontsize=9, color=OCEAN, fontweight="bold")
    ax.text(0.02, 0.41, "COMMUNICATION ET SOL  —  sat7/relay.py, sat7/pipeline.py (ground_decode)",
            fontsize=9, color=ORANGE, fontweight="bold")
    top = [("Grande image\nI ∈ R^(H×W×C)\néq. 1", "scenes.py"),
           ("Prétraitement\nE_pre", "pipeline.py\npreprocess"),
           ("Découpage\n512 px, 20 %\néq. 2 · E_tile", "sahi.py\nslice_grid"),
           ("SAHI + YOLO\nsur chaque tuile\néq. 3-4 · E_det", "sahi.py · detectors.py"),
           ("Coordonnées\nglobales\néq. 5", "sahi.py\nto_global"),
           ("Fusion\nNMS / NMM\néq. 6 · E_fusion", "sahi.py\nfuse"),
           ("Enregistrements\n+ priorités P0-P3\néq. 7, Tab. I", "semantic.py\nbuild_packet")]
    w, h, y = 0.118, 0.20, 0.62
    for i, (t, f) in enumerate(top):
        x = 0.02 + i * 0.139
        box(ax, x, y, w, h, t, fs=8)
        ax.text(x + w / 2, y - 0.012, f, ha="center", va="top", fontsize=7, color=GREY,
                family="monospace")
        if i:
            arrow(ax, (x - 0.021, y + h / 2), (x, y + h / 2))
    bot = [("Paquet sémantique\n(octets réels)\nD_tx", "semantic.py\nSemanticPacket.encode"),
           ("Choix du chemin\nmin J = λE·E + λT·T\néq. 16, 25, 26", "relay.py\nNetwork.send"),
           ("Liaison directe\nprincipal → sol\néq. 12-14, 23", "relay.py\nLinkSchedule"),
           ("Relais ISL\nprincipal → relais → sol\néq. 15, 24, 29", "relay.py\nisl_windows"),
           ("Station sol Sfax\ndécodage\néq. 28", "pipeline.py\nground_decode"),
           ("Évaluation\nDR, rappel, IP,\nénergie, latence", "metrics.py · energy.py")]
    pos = [(0.02, 0.17), (0.19, 0.17), (0.37, 0.25), (0.37, 0.085), (0.58, 0.17), (0.77, 0.17)]
    for (t, f), (x, yy) in zip(bot, pos):
        box(ax, x, yy, 0.15, 0.14, t, ec=ORANGE, fs=7.8)
        ax.text(x + 0.075, yy - 0.012, f, ha="center", va="top", fontsize=6.6, color=GREY,
                family="monospace")
    xs = 0.02 + 6 * 0.139 + w / 2                     # route around the boxes, not through them
    ax.plot([xs, xs, 0.095], [y - 0.075, 0.475, 0.475], color=OCEAN, lw=1.3)
    arrow(ax, (0.095, 0.475), (0.095, 0.31), color=OCEAN, lw=1.3)
    arrow(ax, (0.17, 0.24), (0.19, 0.24))
    arrow(ax, (0.34, 0.26), (0.37, 0.32))
    arrow(ax, (0.34, 0.22), (0.37, 0.155))
    arrow(ax, (0.52, 0.32), (0.58, 0.26))
    arrow(ax, (0.52, 0.155), (0.58, 0.20))
    arrow(ax, (0.73, 0.24), (0.77, 0.24))
    ax.text(0.98, 0.91, "énergie de calcul : E_proc = Σ P_k·T_k  (éq. 9-11)", ha="right",
            fontsize=8.5, color=OCEAN)
    fig.savefig(OUT / "pipeline_flow.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- 2. file relations (paper)
def relations_paper():
    fig, ax = canvas(12, 6.4)
    layers = [(0.86, "Script d'expérience", OCEAN), (0.66, "Orchestration", OCEAN),
              (0.42, "Briques du papier", OCEAN), (0.20, "Socle existant réutilisé", GREY),
              (0.03, "Sorties", ORANGE)]
    for y, name, c in layers:
        ax.text(0.005, y + 0.06, name, fontsize=8.5, color=c, fontweight="bold", rotation=0)
    nodes = {
        "scripts/paper_b0_b4.py": (0.36, 0.86, 0.28),
        "pipeline.py": (0.40, 0.66, 0.20),
        "scenes.py": (0.13, 0.42, 0.10), "detectors.py": (0.25, 0.42, 0.10),
        "sahi.py": (0.37, 0.42, 0.10), "semantic.py": (0.49, 0.42, 0.10),
        "metrics.py": (0.61, 0.42, 0.10), "energy.py": (0.73, 0.42, 0.10),
        "relay.py": (0.85, 0.42, 0.10),
        "synthetic.py": (0.13, 0.20, 0.10), "prefilter.py": (0.25, 0.20, 0.10),
        "rle.py": (0.19, 0.20 - 0.0, 0.0), "orbit.py": (0.85, 0.20, 0.10),
        "results/paper_*.csv|json|png": (0.30, 0.03, 0.26),
        "guide_figures.py → PDF / dashboard": (0.62, 0.03, 0.30),
    }
    nodes.pop("rle.py")
    h = 0.09
    for n, (x, y, w) in nodes.items():
        ec = GREY if y == 0.20 else (ORANGE if y < 0.1 else OCEAN)
        box(ax, x, y, w, h, n, ec=ec, fs=8, bold=n.startswith("scripts") or n == "pipeline.py")

    def c(n, side):
        x, y, w = nodes[n]
        return (x + w / 2, y + (h if side == "top" else 0))

    edges = [("scripts/paper_b0_b4.py", "pipeline.py"), ("scripts/paper_b0_b4.py", "scenes.py"),
             ("scripts/paper_b0_b4.py", "detectors.py"), ("scripts/paper_b0_b4.py", "metrics.py"),
             ("scripts/paper_b0_b4.py", "energy.py"), ("scripts/paper_b0_b4.py", "relay.py"),
             ("pipeline.py", "sahi.py"), ("pipeline.py", "semantic.py"),
             ("detectors.py", "prefilter.py"), ("scenes.py", "synthetic.py"),
             ("relay.py", "orbit.py")]
    for a, b in edges:
        arrow(ax, c(a, "bot"), c(b, "top"), color=OCEAN if a.startswith("scripts") else GREY, lw=1.0)
    for a, b in (("detectors.py", "sahi.py"), ("semantic.py", "sahi.py"), ("metrics.py", "sahi.py"),
                 ("relay.py", "energy.py"), ("scenes.py", "sahi.py")):
        (x1, y1, w1), (x2, y2, w2) = nodes[a], nodes[b]
        a_pt, b_pt = (x1 + w1 / 2, y1), (x2 + w2 / 2 + (0.02 if x1 > x2 else -0.02), y2)
        arrow(ax, a_pt, b_pt, color="#9CA3AF", lw=0.9, rad=0.45 if x1 < x2 else -0.45)
    ax.text(0.43, 0.125, "écrites par paper_b0_b4.py", ha="center", fontsize=7.5, color=ORANGE)
    arrow(ax, (0.56, 0.075), (0.62, 0.075), color=ORANGE)
    ax.text(0.70, 0.80, "flèche = « importe / utilise »\nbleu : appelé par le script\n"
            "gris : dépendance interne\n(courbes : tous utilisent Detection\nde sahi.py ; relay utilise energy)",
            fontsize=8, color=GREY)
    fig.savefig(OUT / "relations_paper.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- 3. file relations (WP chain)
def relations_wp():
    fig, ax = canvas(12, 5.2)
    cols = [("1. Données", ["wp0_build_dataset", "airbus_to_yolo (ancien)"]),
            ("2. Modèles", ["wp1_train_detector", "wp1_export_int8", "wp2_predict_split",
                            "wp2_calibrate", "eval_prefilter", "wp3_train_gate", "wp3_score_tiles"]),
            ("3. Coûts en octets", ["wp4_measure_lod", "wp6_fit_size_model",
                                    "wp7_measure_cheap_products", "contact_windows"]),
            ("4. Simulation", ["wp6_build_catalogue", "simulate_day", "wp6_simulate_real",
                               "wp7_budget_sweep", "wp8_queue_aware", "wp14_gate_signal"]),
            ("5. Vérifications", ["wp5_optimality_gap", "wp5_joint_bound", "wp10_sensitivity",
                                  "wp10b_interaction", "wp11_integration_demo", "wp12_seams",
                                  "wp12b_cut_recall", "wp13_recompress_check"]),
            ("6. Rapports", ["build_report", "build_overview", "build_phase2_guide"])]
    xw = 0.155
    for i, (t, items) in enumerate(cols):
        x = 0.01 + i * 0.165
        ax.text(x + xw / 2, 0.95, t, ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
        for j, it in enumerate(items):
            y = 0.84 - j * 0.095
            box(ax, x, y, xw, 0.075, it + ".py" if "(" not in it else it, fs=7.4,
                ec=OCEAN if i < 5 else ORANGE)
        if i:
            arrow(ax, (x - 0.022, 0.958), (x + 0.008, 0.958), color=NAVY, lw=1.6)
    ax.text(0.5, 0.02, "scripts : code/legacy_wp/scripts/ (sauf wp0, wp1, wp2 → code/scripts/) · "
            "modules : legacy_wp/wplib + sat7 · sorties : code/results/",
            ha="center", fontsize=8, color=GREY)
    fig.savefig(OUT / "relations_wp.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- dashboard chrome
def chrome(title, active):
    fig = plt.figure(figsize=(13, 7.6), facecolor=LIGHT)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")
    bg.add_patch(Rectangle((0, 0), 0.17, 1, fc=NAVY))
    bg.text(0.015, 0.95, "SAT7 · Downlink", color="white", fontsize=12, fontweight="bold")
    bg.text(0.015, 0.925, "tableau de bord — station sol", color="#9FB3C8", fontsize=8)
    pages = ["Vue d'ensemble", "Communication", "Compromis & scène", "Ablations", "Hypothèses"]
    for i, p in enumerate(pages):
        y = 0.85 - i * 0.045
        if p == active:
            bg.add_patch(Rectangle((0.008, y - 0.012), 0.154, 0.036, fc=OCEAN))
        bg.text(0.02, y, ("▸ " if p == active else "   ") + p, color="white", fontsize=9.5)
    bg.text(0.015, 0.56, "FILTRES", color="#9FB3C8", fontsize=8, fontweight="bold")
    for i, (k, v) in enumerate([("Source", "SIM (synthétique)  ▾"), ("Configuration", "B4  ▾"),
                                ("Liaison", "cubesat_sband  ▾")]):
        y = 0.52 - i * 0.075
        bg.text(0.015, y, k, color="white", fontsize=8)
        bg.add_patch(Rectangle((0.015, y - 0.04), 0.14, 0.03, fc="#13325C", ec="#3A5F8F"))
        bg.text(0.022, y - 0.031, v, color="white", fontsize=8)
    bg.text(0.015, 0.27, "Seuil de confiance : 0,25", color="white", fontsize=8)
    bg.add_patch(Rectangle((0.015, 0.245), 0.14, 0.006, fc="#3A5F8F"))
    bg.add_patch(Rectangle((0.015, 0.245), 0.035, 0.006, fc=ORANGE))
    bg.scatter([0.05], [0.248], s=60, color=ORANGE, zorder=3)
    bg.text(0.015, 0.06, "lit : code/results/paper_*.csv\nmis à jour à chaque\nexécution de paper_b0_b4.py",
            color="#9FB3C8", fontsize=7)
    bg.text(0.19, 0.95, title, fontsize=15, color=NAVY, fontweight="bold")
    bg.add_patch(FancyBboxPatch((0.19, 0.885), 0.79, 0.04, boxstyle="round,pad=0,rounding_size=0.006",
                                fc="#FDF1E6", ec=ORANGE))
    bg.text(0.20, 0.898, "⚠  Résultats SIM : scènes synthétiques + détecteur classique de secours. "
            "Ne pas citer comme résultats YOLO. Les puissances sont des HYPOTHÈSES.",
            fontsize=8.5, color="#8A4A0F")
    return fig, bg


def panel(fig, rect, title):
    bg = fig.axes[0]
    x, y, w, h = rect
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                fc=CARD, ec="#DDE3EA"))
    bg.text(x + 0.01, y + h - 0.03, title, fontsize=9.5, color=NAVY, fontweight="bold")
    ax = fig.add_axes([x + 0.045, y + 0.06, w - 0.065, h - 0.13])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.25)
    return ax


def kpi(bg, x, y, label, value, sub, color=NAVY):
    bg.add_patch(FancyBboxPatch((x, y), 0.148, 0.115, boxstyle="round,pad=0,rounding_size=0.008",
                                fc=CARD, ec="#DDE3EA"))
    bg.text(x + 0.01, y + 0.083, label, fontsize=8, color=GREY)
    bg.text(x + 0.01, y + 0.035, value, fontsize=17, color=color, fontweight="bold")
    bg.text(x + 0.01, y + 0.012, sub, fontsize=7, color=GREY)


# ---------------------------------------------------------------- 4. dashboard pages
def dash_overview(tab, abl):
    fig, bg = chrome("Vue d'ensemble — campagne B0 → B4", "Vue d'ensemble")
    b4 = tab.set_index("config").loc["B4"]
    for i, (lab, val, sub, c) in enumerate([
            ("Réduction des données", f"×{b4.reduction_x:,.0f}".replace(",", " "), "D_raw / D_tx  (éq. 27)", OCEAN),
            ("Rappel (navires)", f"{b4.recall:.2f}", f"F1 {b4.F1:.2f} · mAP50 {b4.mAP50:.2f}", GREEN),
            ("Énergie économisée", f"{b4.ES:.1%}".replace(".", ","), "ES vs B0  (éq. 17)", GREEN),
            ("Latence médiane", f"{b4.T_total_med_h:.2f} h".replace(".", ","), "T_total  (éq. 28)", NAVY),
            ("Part via relais", f"{b4.relay_share:.0%}", "paquets passés par l'ISL", PURPLE)]):
        kpi(bg, 0.19 + i * 0.159, 0.745, lab, val, sub, c)
    ax = panel(fig, (0.19, 0.38, 0.385, 0.345), "Données transmises par configuration (log)")
    ax.bar(tab.config, tab.D_tx_MB, color=[GREY, OCEAN, OCEAN, GREEN, GREEN])
    ax.set_yscale("log")
    ax.set_ylabel("D_tx (Mo)", fontsize=8)
    for i, v in enumerate(tab.D_tx_MB):
        ax.text(i, v * 1.3, f"{v:.3g}", ha="center", fontsize=7)
    ax = panel(fig, (0.595, 0.38, 0.385, 0.345), "Qualité : rappel, F1, préservation de l'information")
    x = np.arange(len(tab))
    for k, (col, c) in enumerate((("recall", OCEAN), ("F1", ORANGE), ("IP", GREEN))):
        ax.bar(x + (k - 1) * 0.26, tab[col], 0.26, color=c, label=col)
    ax.set_xticks(x, tab.config)
    ax.set_ylim(0, 1.3)
    ax.legend(["rappel", "F1", "IP"], fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax = panel(fig, (0.19, 0.03, 0.385, 0.33), "Énergie : calcul à bord vs communication")
    ax.bar(tab.config, tab.E_proc_J, color=OCEAN, label="E_proc (calcul)")
    ax.bar(tab.config, tab.E_comm_J, bottom=tab.E_proc_J, color=ORANGE, label="E_comm")
    ax.set_yscale("log")
    ax.set_ylabel("J (log)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    bgx = fig.axes[0]
    bgx.add_patch(FancyBboxPatch((0.595, 0.03), 0.385, 0.33, boxstyle="round,pad=0,rounding_size=0.008",
                                 fc=CARD, ec="#DDE3EA"))
    bgx.text(0.605, 0.33, "Ablations (VIII-G) — écart vs B4 complet", fontsize=9.5, color=NAVY,
             fontweight="bold")
    full = abl.iloc[0]
    for i, (_, r) in enumerate(abl.iloc[1:].iterrows()):
        y = 0.285 - i * 0.04
        bgx.text(0.607, y, FR_ABL.get(r.config, r.config), fontsize=8, color=NAVY)
        for j, (col, nm) in enumerate((("recall", "rappel"), ("D_tx_MB", "octets"), ("T_total_med_h", "latence"))):
            d = r[col] - full[col]
            rel = d / full[col] if full[col] else 0
            good = (d >= 0) if col == "recall" else (d <= 0)
            txt = f"{nm} {rel:+.0%}" if abs(rel) > 0.005 else f"{nm} ="
            bgx.text(0.78 + j * 0.067, y, txt, fontsize=7.4,
                     color=GREY if abs(rel) <= 0.005 else (GREEN if good else RED))
    fig.savefig(OUT / "dashboard_1_vue_ensemble.png", dpi=150, facecolor=LIGHT)
    plt.close(fig)


def dash_comms(pk, windows):
    fig, bg = chrome("Communication — passages, relais et paquets (B4)", "Communication")
    w_direct, w_isl, w_relay = windows
    ax = panel(fig, (0.19, 0.42, 0.79, 0.44), "Chronologie sur 48 h : fenêtres de liaison et trajet de chaque paquet")
    lanes = {"Passages principal → Sfax": (3, w_direct, OCEAN),
             "Fenêtres ISL principal ↔ relais": (2, w_isl, PURPLE),
             "Passages relais → Sfax": (1, w_relay, ORANGE)}
    for name, (yv, ws, c) in lanes.items():
        for w in ws:
            ax.add_patch(Rectangle((w.start / 3600, yv - 0.3), max((w.end - w.start) / 3600, 0.08), 0.6,
                                   fc=c, alpha=0.75, lw=0))
    for _, r in pk.iterrows():
        c = PURPLE if r.path == "relay" else OCEAN
        ax.plot([r.t_capture_h, r.t_delivered_h], [0.25, 0.25], color=c, lw=1.2, alpha=0.8)
        ax.scatter([r.t_capture_h], [0.25], s=10, color=GREY, zorder=3)
        ax.scatter([r.t_delivered_h], [0.25], s=16, color=c, zorder=3, marker=">")
    ax.set_yticks([])
    for yv, txt in zip([3, 2, 1, 0.25], list(lanes) + ["Paquets : capture → livraison"]):
        ax.text(0.2, yv + 0.36, txt, fontsize=7.5, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 36)
    ax.set_ylim(-0.1, 3.55)
    ax.set_xlabel("heures depuis le début de la journée", fontsize=8)
    ax = panel(fig, (0.19, 0.03, 0.385, 0.37), "Latence par paquet selon le chemin choisi")
    for p, c in (("direct", OCEAN), ("relay", PURPLE)):
        d = pk[pk.path == p]
        ax.scatter(d.t_capture_h, d.T_total / 3600, s=26, color=c, label=f"{p} ({len(d)})")
    ax.axhline(6, color=RED, ls="--", lw=1)
    ax.text(0.3, 6.15, "échéance 6 h", fontsize=7, color=RED)
    ax.set_xlabel("heure de capture (h)", fontsize=8)
    ax.set_ylabel("T_total (h)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    bgx = fig.axes[0]
    bgx.add_patch(FancyBboxPatch((0.595, 0.03), 0.385, 0.37, boxstyle="round,pad=0,rounding_size=0.008",
                                 fc=CARD, ec="#DDE3EA"))
    bgx.text(0.605, 0.37, "Derniers paquets", fontsize=9.5, color=NAVY, fontweight="bold")
    cols = ["scène", "capture", "chemin", "octets", "E_comm (J)", "latence"]
    for j, cname in enumerate(cols):
        bgx.text(0.607 + j * 0.062, 0.335, cname, fontsize=7.5, color=GREY, fontweight="bold")
    for i, (_, r) in enumerate(pk.sort_values("t_capture_h").tail(9).iterrows()):
        y = 0.305 - i * 0.03
        vals = [f"{int(r.scene)}", f"{r.t_capture_h:.1f} h", r.path, f"{int(r.bytes):,}".replace(",", " "),
                f"{r.E_comm:.2f}", f"{r.T_total / 3600:.2f} h"]
        for j, v in enumerate(vals):
            bgx.text(0.607 + j * 0.062, y, v, fontsize=7.5,
                     color=(PURPLE if v == "relay" else NAVY))
    fig.savefig(OUT / "dashboard_2_communication.png", dpi=150, facecolor=LIGHT)
    plt.close(fig)


def dash_tradeoff(sw, tab):
    fig, bg = chrome("Compromis & scène — choisir le seuil, voir les priorités", "Compromis & scène")
    img, gts = synthetic_scene(2048, 18, seed=7)
    pipe = OnboardPipeline(ClassicDetector(), PipelineConfig())
    out = pipe.run(img, 0, 0.0)
    ax = fig.add_axes([0.19, 0.03, 0.40, 0.83])
    bg.add_patch(FancyBboxPatch((0.185, 0.02), 0.41, 0.86, boxstyle="round,pad=0,rounding_size=0.008",
                                fc=CARD, ec="#DDE3EA"))
    ax.imshow(img[..., ::-1])
    for (x, y, w, h) in slice_grid(2048, 2048, 512, 0.2):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="white", lw=0.4, alpha=0.35))
    pol = pipe.cfg.policy
    for d in out.detections:
        lvl = pol.priority(d.p)
        pad = 18
        ax.add_patch(Rectangle((d.x - pad, d.y - pad), d.w + 2 * pad, d.h + 2 * pad, fill=False,
                               ec=PRIO.get(lvl, GREY), lw=1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Scène 2048 × 2048 · grille SAHI 512 px / 20 % · cadres = priorité",
                 fontsize=9, color=NAVY)
    for i, (lvl, txt) in enumerate(((1, "P1 métadonnées"), (2, "P2 + ROI"), (3, "P3 + ROI + contexte"))):
        n = sum(pol.priority(d.p) == lvl for d in out.detections)
        bg.text(0.20 + i * 0.13, 0.035, f"■ {txt} ({n})", color=PRIO[lvl], fontsize=8, fontweight="bold")
    ax = panel(fig, (0.61, 0.46, 0.37, 0.40), "(32) Octets envoyés vs rappel")
    ax.plot(sw.D_tx_MB, sw.recall, "o-", color=OCEAN)
    sel = sw.iloc[(sw.det_thr - 0.25).abs().argmin()]
    ax.scatter([sel.D_tx_MB], [sel.recall], s=120, facecolor="none", ec=ORANGE, lw=2, zorder=4)
    ax.annotate("seuil 0,25", (sel.D_tx_MB, sel.recall), xytext=(-60, -22), textcoords="offset points",
                fontsize=7.5, color=ORANGE)
    ax.set_xlabel("D_tx (Mo)", fontsize=8)
    ax.set_ylabel("rappel", fontsize=8)
    ax = panel(fig, (0.61, 0.03, 0.37, 0.41), "(34) Octets envoyés vs préservation de l'information")
    ax.plot(sw.D_tx_MB, sw.IP, "o-", color=GREEN)
    ax.scatter([sel.D_tx_MB], [sel.IP], s=120, facecolor="none", ec=ORANGE, lw=2, zorder=4)
    best = sw.iloc[(sw.det_thr - 0.55).abs().argmin()]
    ax.annotate("optimum éq. 20 (0,55)", (best.D_tx_MB, best.IP), xytext=(-20, -30),
                textcoords="offset points", fontsize=7.5, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax.set_xlabel("D_tx (Mo)", fontsize=8)
    ax.set_ylabel("IP", fontsize=8)
    fig.savefig(OUT / "dashboard_3_compromis.png", dpi=150, facecolor=LIGHT)
    plt.close(fig)


# ---------------------------------------------------------------- 5. the whole project
def project_map():
    fig, ax = canvas(13, 7.4)
    cols = [(0.01, "1. DONNÉES"), (0.205, "2. MODÈLES"), (0.40, "3. MESURES"),
            (0.595, "4. SIMULATIONS"), (0.79, "5. SORTIES")]
    for x, t in cols:
        ax.text(x + 0.09, 0.965, t, ha="center", fontsize=10, fontweight="bold", color=NAVY)
    w, h = 0.18, 0.075

    def n(x, y, text, ec=OCEAN, fc=CARD, bold=False):
        box(ax, x, y, w, h, text, ec=ec, fc=fc, fs=7.4, bold=bold)
        return (x, y)

    zipf = n(0.01, 0.82, "Archive Airbus (Kaggle)\n192 556 tuiles 768 px", ec=GREY)
    ds = n(0.01, 0.64, "wp0_build_dataset.py\n→ data/yolo_ships\n(train / val / test sans fuite)", bold=True)
    orb = n(0.01, 0.30, "sat7/orbit.py\norbite SGP4 + station Sfax", ec=GREY)
    yolo = n(0.205, 0.82, "wp1_train_detector.py\n→ best.pt (YOLOv8n)", bold=True)
    pred = n(0.205, 0.68, "wp1_predictions.csv\nwp1_gt_matched.csv", ec=ORANGE)
    cal = n(0.205, 0.54, "wp2_calibrate.py\n→ seuils de confiance")
    gate = n(0.205, 0.40, "wp3_train_gate.py\n→ gate.pt (47k param.)")
    lod = n(0.40, 0.82, "wp4_measure_lod.py (lit data/)\n→ wp4_lod_sizes.csv")
    sizem = n(0.40, 0.68, "wp6_fit_size_model.py\n→ wp6_size_model.json")
    cat = n(0.40, 0.54, "wp6_build_catalogue.py\n→ wp6_tiles.csv, wp6_ships.csv", bold=True)
    cw = n(0.40, 0.30, "contact_windows.py\n→ passes.csv")
    rw = n(0.595, 0.68, "legacy_wp/wplib/real_workload.py\njournée réelle à partir\ndu catalogue",
           ec=GREY)
    sim = n(0.595, 0.54, "wp6_simulate_real.py\n★ 557×, rappel 0,685", fc="#EAF6EF", ec=GREEN, bold=True)
    more = n(0.595, 0.40, "wp5, wp7, wp8, wp10,\nwp11, wp12, wp13, wp14\nvérifications + améliorations")
    paper = n(0.595, 0.16, "paper_b0_b4.py (charge best.pt)\nSAHI · P0-P3 · énergie · relais\n(pipeline du papier)",
              fc="#EAF2FA", ec=OCEAN, bold=True)
    res = n(0.79, 0.54, "code/results/\n*.csv · *.json · *.png", ec=ORANGE)
    pdf = n(0.79, 0.36, "build_report.py, build_*_guide.py\n→ PDF du dossier paper/", ec=ORANGE)
    dash = n(0.79, 0.20, "dashboard/app.py\n(Streamlit, lit results/)", ec=ORANGE, fc=LIGHT)

    def link(a, b, c=GREY, rad=0.0):
        arrow(ax, (a[0] + w, a[1] + h / 2), (b[0], b[1] + h / 2), color=c, rad=rad)

    def down(a, b, c=GREY):
        arrow(ax, (a[0] + w / 2, a[1]), (b[0] + w / 2, b[1] + h), color=c)

    down(zipf, ds)
    link(ds, yolo, OCEAN)
    down(yolo, pred, OCEAN)
    link(ds, gate)
    down(pred, cal)
    down(lod, sizem)
    link(pred, cat, OCEAN, rad=-0.1)
    link(sizem, rw)
    link(cat, rw, OCEAN)
    link(orb, cw)
    down(rw, sim, GREEN)
    down(sim, more)
    link(cw, sim, rad=0.15)
    link(orb, paper, GREY, rad=0.1)
    link(sim, res, GREEN)
    link(more, res)
    link(paper, res, OCEAN, rad=-0.2)
    down(res, pdf, ORANGE)
    down(pdf, dash, ORANGE)
    ax.text(0.5, 0.04, "en gras : les étapes clés · vert : le résultat principal mesuré · bleu clair : "
            "le pipeline du papier · sat7/ = la bibliothèque partagée", ha="center",
            fontsize=8.5, color=GREY)
    fig.savefig(OUT / "project_map.png", dpi=170)
    plt.close(fig)


def wp_onboard():
    fig, ax = canvas(12, 2.7)
    steps = [("Tuile\n768 × 768 px", "image du satellite", GREY),
             ("Gate (CNN)\n47 000 paramètres", "0,15 ms · mer vide ?\n→ on saute", OCEAN),
             ("Détecteur YOLOv8n\n768 px", "11 ms · boîtes +\nconfiance", OCEAN),
             ("Niveau de détail\n(LoD)", "sûr → 40 o\nincertain → ~900 o\nzone → ~2,5 ko\ncôte → tuile", ORANGE),
             ("Ordonnanceur\nvalue-greedy", "valeur / octet,\nsur plusieurs passages", GREEN),
             ("Station sol\nSfax", "5 passages / 24 h", GREY)]
    for i, (t, sub, c) in enumerate(steps):
        x = 0.01 + i * 0.165
        box(ax, x, 0.52, 0.145, 0.42, t, ec=c, fs=9, bold=True)
        ax.text(x + 0.0725, 0.46, sub, ha="center", va="top", fontsize=8, color=GREY)
        if i:
            arrow(ax, (x - 0.02, 0.73), (x, 0.73), color=NAVY, lw=1.6)
    fig.savefig(OUT / "wp_onboard.png", dpi=170)
    plt.close(fig)


def sat7_graph():
    fig, ax = canvas(12, 5.0)
    w, h = 0.105, 0.11
    pos = {"rle": (0.06, 0.08), "synthetic": (0.06, 0.35), "prefilter": (0.20, 0.35),
           "dedup": (0.20, 0.08), "orbit": (0.62, 0.08), "scheduler": (0.62, 0.35),
           "optimum": (0.62, 0.62), "real_workload": (0.75, 0.62), "sahi": (0.34, 0.35),
           "detectors": (0.14, 0.62), "scenes": (0.02, 0.62), "semantic": (0.48, 0.62),
           "metrics": (0.20, 0.86), "pipeline": (0.34, 0.86), "energy": (0.88, 0.35),
           "relay": (0.88, 0.62)}
    paper = {"sahi", "detectors", "scenes", "semantic", "metrics", "pipeline", "energy", "relay"}
    archived = {"scheduler", "optimum", "real_workload"}
    for k, (x, y) in pos.items():
        box(ax, x, y, w, h, ("wplib/" if k in archived else "") + k + ".py",
            ec=OCEAN if k in paper else (ORANGE if k in archived else GREY),
            fs=7 if k in archived else 8,
            fc="#EAF2FA" if k in paper else CARD)
    deps = [("detectors", "prefilter"), ("detectors", "sahi"), ("metrics", "sahi"),
            ("optimum", "scheduler"), ("pipeline", "sahi"), ("pipeline", "semantic"),
            ("prefilter", "rle"), ("real_workload", "scheduler"), ("relay", "energy"),
            ("relay", "orbit"), ("scenes", "sahi"), ("scenes", "synthetic"),
            ("scheduler", "orbit"), ("semantic", "sahi"), ("synthetic", "rle")]
    for a, b in deps:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        p1 = (x1 + w / 2, y1 if y1 > y2 else y1 + h)
        p2 = (x2 + w / 2, y2 + h if y1 > y2 else y2)
        if y1 == y2:
            p1, p2 = (x1 + (0 if x1 > x2 else w), y1 + h / 2), (x2 + (w if x1 > x2 else 0), y2 + h / 2)
        arrow(ax, p1, p2, color=OCEAN if a in paper else GREY, lw=1.0)
    ax.text(0.99, 0.95, "bleu : modules du papier (sat7)\ngris : socle partagé (sat7)\n"
            "orange : archivés (legacy_wp/wplib)\nflèche A → B : A importe B", ha="right", va="top", fontsize=8.5, color=GREY)
    fig.savefig(OUT / "sat7_graph.png", dpi=170)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    project_map()
    wp_onboard()
    sat7_graph()
    tab = pd.read_csv(RES / "paper_b0_b4_sim.csv")
    abl = pd.read_csv(RES / "paper_ablations_sim.csv")
    sw = pd.read_csv(RES / "paper_tradeoff_sim.csv")
    pk = pd.read_csv(RES / "paper_packets_sim.csv")
    epoch = datetime(2026, 9, 18, tzinfo=timezone.utc)
    prim, rcfg = OrbitConfig(epoch=epoch), RelayConfig()
    rel = relay_orbit(prim, rcfg)
    link, gs = LINK_PRESETS["cubesat_sband"], GroundStation()
    windows = (ground_windows(prim, gs, link, epoch, 48), isl_windows(prim, rel, epoch, 48, rcfg),
               ground_windows(rel, gs, link, epoch, 48))
    pipeline_flow()
    relations_paper()
    relations_wp()
    dash_overview(tab, abl)
    dash_comms(pk, windows)
    dash_tradeoff(sw, tab)
    print("saved", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
