"""SAT7 dashboard: the ground station's view of the semantic-downlink pipeline.

It shows what scripts/paper_b0_b4.py and the WP scripts wrote in results/ and never recomputes
a result. The Simulation page launches paper_b0_b4.py itself (dashboard/runner.py) into
results/live/<run>/; every other page can then display that run, read from its files.

    cd code
    .venv/Scripts/streamlit run dashboard/app.py          # -> http://localhost:8501
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import runner
from loaders import (COMPARE_METRICS, LIVE_PREFIX, PRIORITY_FR, STRATEGY_FR, ablation_deltas,
                     available_sources, compare_campaigns, live_sources, load_imported,
                     load_measured, load_paper, run_settings)
from sat7.orbit import LINK_PRESETS, STATION_PRESETS
from sat7.sahi import slice_grid

NAVY, OCEAN, ORANGE, GREEN, RED, PURPLE, GREY = ("#0B2545", "#1B6CA8", "#E07A1F", "#2E8B57",
                                                 "#C0392B", "#7D3C98", "#6B7280")
PRIO_COLOR = {1: GREEN, 2: ORANGE, 3: RED}
CONFIG_COLOR = {"B0": GREY, "B1": OCEAN, "B2": OCEAN, "B3": GREEN, "B4": GREEN, "B4+J": PURPLE}
PAGES = ["Vue d'ensemble", "Communication", "Compromis", "Scène", "Ablations", "Simulation",
         "Comparer", "Mesures réelles (fusion)", "Résultats mesurés (WP)", "Hypothèses"]
LIVE_IN_SIDEBAR = 10                                          # most recent live runs listed


def fr(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def pct(v: float, d: int = 1) -> str:
    return f"{100 * v:.{d}f} %".replace(".", ",")


@st.cache_data
def paper(source: str):
    return load_paper(source)


@st.cache_data
def measured():
    return load_measured()


def source_name(src: str) -> str:
    """How a source reads in the selectors: REAL, SIM, or LIVE · date · label."""
    if not src.startswith(LIVE_PREFIX):
        return src
    rid = src[len(LIVE_PREFIX):]
    stamp = time.strftime("%d/%m %H:%M:%S", time.strptime(rid.split("_")[0], "%Y%m%d-%H%M%S"))
    return f"LIVE · {stamp} · {paper(src).label}"


def banner(res) -> None:
    if res.label == "SIM":
        why = [w for w, on in (("scènes synthétiques", res.meta.get("source") == "synthetic"),
                               ("détecteur classique de secours",
                                "classic" in str(res.meta.get("detector", "classic"))))
               if on] or ["pas de détecteur YOLO sur de vraies scènes"]
        st.warning(f"**Résultats SIM** : {' et '.join(why)}. "
                   "Ne pas les citer comme résultats YOLO. Les puissances sont des hypothèses.",
                   icon="⚠️")
    else:
        st.success("**Résultats REAL** : tuiles Airbus et détecteur YOLO entraîné. "
                   "Les puissances restent des hypothèses.", icon="✅")
    if res.live_id:
        m = res.meta
        st.caption(f"Run lancé depuis « Simulation » : `results/live/{res.live_id}/` · "
                   f"{m['scenes']} scènes {m['source']}, détecteur {m['detector']}, tuiles "
                   f"{m['sahi']['tile']} px, seuil {m['sahi']['conf_thr']}.")


# ------------------------------------------------------------------ pages
def page_overview(res) -> None:
    st.header("Vue d'ensemble — campagne B0 → B4")
    banner(res)
    c = res.campaign.set_index("config")
    b4, b0 = c.loc["B4"], c.loc["B0"]
    k = st.columns(5)
    k[0].metric("Réduction des données", f"×{fr(b4.reduction_x, 0)}", help="D_raw / D_tx (éq. 27)")
    k[1].metric("Rappel", fr(b4.recall), help=f"F1 {fr(b4.F1)} · mAP50 {fr(b4.mAP50)}")
    k[2].metric("Énergie économisée", pct(b4.ES), help="ES = 1 − E_B4 / E_B0 (éq. 17)")
    k[3].metric("Latence médiane", f"{fr(b4.T_total_med_h)} h",
                delta=f"{fr(b4.T_total_med_h - b0.T_total_med_h)} h vs B0", delta_color="inverse")
    k[4].metric("Part via relais", pct(b4.relay_share, 0))

    left, right = st.columns(2)
    fig = px.bar(res.campaign, x="config", y="D_tx_MB", log_y=True, text_auto=".3g",
                 labels={"config": "", "D_tx_MB": "D_tx (Mo, log)"},
                 color="config", color_discrete_map=CONFIG_COLOR)
    fig.update_layout(title="Données transmises", showlegend=False, height=360)
    left.plotly_chart(fig, width="stretch")

    q = res.campaign.melt(id_vars="config", value_vars=["recall", "F1", "IP"],
                          var_name="mesure", value_name="valeur")
    q["mesure"] = q.mesure.map({"recall": "rappel", "F1": "F1", "IP": "préservation (IP)"})
    fig = px.bar(q, x="config", y="valeur", color="mesure", barmode="group",
                 color_discrete_sequence=[OCEAN, ORANGE, GREEN], labels={"config": ""})
    fig.update_layout(title="Qualité", yaxis_range=[0, 1.05], height=360, legend_title="")
    right.plotly_chart(fig, width="stretch")

    e = res.campaign.melt(id_vars="config", value_vars=["E_proc_J", "E_comm_J"],
                          var_name="poste", value_name="J")
    e["poste"] = e.poste.map({"E_proc_J": "calcul à bord", "E_comm_J": "communication"})
    fig = px.bar(e, x="config", y="J", color="poste", log_y=True,
                 color_discrete_sequence=[OCEAN, ORANGE], labels={"config": ""})
    fig.update_layout(title="Énergie (J, log)", height=340, legend_title="")
    left.plotly_chart(fig, width="stretch")

    t = res.campaign[["config", "D_tx_MB", "reduction_x", "recall", "F1", "IP", "E_total_J", "ES",
                      "T_total_med_h", "relay_share"]].rename(columns={
                          "D_tx_MB": "Mo envoyés", "reduction_x": "réduction ×", "recall": "rappel",
                          "E_total_J": "énergie (J)", "T_total_med_h": "latence méd. (h)",
                          "relay_share": "part relais"})
    right.markdown("**Tableau B0 → B4**")
    right.dataframe(t.round(3), hide_index=True, width="stretch")


def page_comms(res) -> None:
    st.header("Communication — passages, relais et paquets (B4)")
    banner(res)
    pk, links = res.packets, res.links
    hz = float(res.meta.get("horizon_hours", 48))
    lanes = {"direct": ("Passages principal → Sfax", 3, OCEAN),
             "isl": ("Fenêtres ISL principal ↔ relais", 2, PURPLE),
             "relay_gs": ("Passages relais → Sfax", 1, ORANGE)}
    fig = go.Figure()
    for kind, (name, y, color) in lanes.items():
        for _, w in links[links.link == kind].iterrows():
            fig.add_shape(type="rect", x0=w.start_h, x1=max(w.end_h, w.start_h + 0.06),
                          y0=y - 0.3, y1=y + 0.3, fillcolor=color, opacity=0.75, line_width=0)
    for path, color in (("direct", OCEAN), ("relay", PURPLE)):
        d = pk[pk.path == path]
        xs, ys = [], []
        for _, r in d.iterrows():
            xs += [r.t_capture_h, r.t_delivered_h, None]
            ys += [0.2, 0.2, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2),
                                 name=f"paquet via {path}", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d.t_delivered_h, y=[0.2] * len(d), mode="markers",
                                 marker=dict(color=color, symbol="triangle-right", size=10),
                                 showlegend=False,
                                 customdata=np.stack([d.scene, d.bytes, d.T_total / 3600], axis=-1),
                                 hovertemplate="scène %{customdata[0]}<br>%{customdata[1]} octets"
                                               "<br>latence %{customdata[2]:.2f} h<extra></extra>"))
    fig.add_trace(go.Scatter(x=pk.t_capture_h, y=[0.2] * len(pk), mode="markers",
                             marker=dict(color=GREY, size=6), name="capture"))
    fig.update_yaxes(tickvals=[3, 2, 1, 0.2],
                     ticktext=[v[0] for v in lanes.values()] + ["Paquets"], range=[-0.3, 3.5])
    fig.update_xaxes(title="heures depuis le début de la journée", range=[0, min(hz, 36)])
    fig.update_layout(title="Chronologie : fenêtres de liaison et trajet de chaque paquet",
                      height=430, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    deadline = float(res.meta.get("deadline_h", 6))
    fig = px.scatter(pk.assign(latence_h=pk.T_total / 3600), x="t_capture_h", y="latence_h",
                     color="path", color_discrete_map={"direct": OCEAN, "relay": PURPLE},
                     labels={"t_capture_h": "heure de capture (h)", "latence_h": "T_total (h)",
                             "path": "chemin"})
    fig.add_hline(y=deadline, line_dash="dash", line_color=RED,
                  annotation_text=f"échéance {fr(deadline, 0)} h")
    fig.update_layout(title="Latence par paquet", height=380)
    left.plotly_chart(fig, width="stretch")

    summary = pk.groupby("path").agg(paquets=("scene", "size"), octets=("bytes", "sum"),
                                     E_comm_J=("E_comm", "sum"),
                                     latence_med_h=("T_total", lambda x: x.median() / 3600))
    summary.index.name = "chemin"
    right.markdown("**Par chemin**")
    right.dataframe(summary.round(3), width="stretch")
    right.markdown("**Paquets**")
    right.dataframe(pk[["scene", "t_capture_h", "path", "bytes", "E_comm", "T_total"]]
                    .assign(T_total=lambda d: d.T_total / 3600)
                    .rename(columns={"t_capture_h": "capture (h)", "path": "chemin",
                                     "bytes": "octets", "E_comm": "E_comm (J)",
                                     "T_total": "latence (h)"})
                    .sort_values("capture (h)").round(3), hide_index=True, width="stretch",
                    height=230)


def page_tradeoff(res) -> None:
    st.header("Compromis — choisir le seuil de confiance")
    banner(res)
    sw = res.sweep
    thr = st.select_slider("Seuil de confiance du détecteur", options=list(sw.det_thr),
                           value=min(sw.det_thr, key=lambda t: abs(t - 0.25)))
    row = sw[sw.det_thr == thr].iloc[0]
    best = res.meta.get("best_operating_point")
    k = st.columns(5)
    k[0].metric("Mo envoyés", fr(row.D_tx_MB, 3))
    k[1].metric("Rappel", fr(row.recall))
    k[2].metric("Précision", fr(row.precision))
    k[3].metric("Préservation (IP)", fr(row.IP))
    k[4].metric("Énergie (J)", fr(row.E_total_J, 0))
    if best:
        obj = res.meta["objective"]
        st.info(f"Meilleur seuil sous l'éq. 20 (F1 ≥ {fr(obj['a_min'])}, IP ≥ "
                f"{fr(obj['i_min'])}) : **{fr(best['det_thr'])}**. La section du bas résout "
                "l'éq. 20 objet par objet.")

    def curve(y, title, color):
        fig = px.line(sw, x="D_tx_MB", y=y, markers=True, hover_data={"det_thr": True},
                      labels={"D_tx_MB": "D_tx (Mo)", "recall": "rappel", "IP": "préservation (IP)",
                              "det_thr": "seuil"})
        fig.update_traces(line_color=color)
        fig.add_trace(go.Scatter(x=[row.D_tx_MB], y=[row[y]], mode="markers", name=f"seuil {thr}",
                                 marker=dict(size=18, color="rgba(0,0,0,0)",
                                             line=dict(color=ORANGE, width=3))))
        fig.update_layout(title=title, height=360, showlegend=False)
        return fig

    left, right = st.columns(2)
    left.plotly_chart(curve("recall", "(32) Octets envoyés vs rappel", OCEAN), width="stretch")
    right.plotly_chart(curve("IP", "(34) Octets envoyés vs préservation de l'information", GREEN),
                       width="stretch")
    fig = px.scatter(res.campaign, x="E_total_J", y="T_total_med_h", text="config", log_x=True,
                     labels={"E_total_J": "E_total (J, log)", "T_total_med_h": "latence médiane (h)"})
    fig.update_traces(marker=dict(size=14, color=NAVY), textposition="top right")
    fig.update_layout(title="(33) Énergie vs latence, B0 → B4", height=360)
    st.plotly_chart(fig, width="stretch")

    if res.joint is None:
        return
    st.subheader("Objectif joint (éq. 20-22) résolu objet par objet")
    jm = res.meta.get("joint_eq20", {})
    w = jm.get("weights", {})
    st.caption(f"Chaque détection reçoit un niveau P0-P3. min αE + βD + γT sous précision ≥ A_min "
               f"et information ≥ I_min. {jm.get('objects', '?')} objets ; α = {w.get('alpha')}, "
               f"β = {w.get('beta')}, γ = {w.get('gamma')} (E en J, D en Mo, T en h) ; solveur "
               "repris de oblhdj, unités corrigées.")
    ims = res.joint[res.joint.sweep == "i_min"]
    i_min = st.select_slider("Seuil d'information I_min (éq. 22)", options=list(ims.i_min),
                             value=float(ims.i_min.iloc[0]))
    jr = ims[ims.i_min == i_min].iloc[0]
    k = st.columns(4)
    k[0].metric("Mo envoyés", fr(jr.D_tx_MB, 3))
    k[1].metric("Information (éq. 22)", fr(jr.information_eq22))
    k[2].metric("Énergie (J)", fr(jr.E_total_J, 0))
    k[3].metric("Niveaux P1 / P2 / P3", f"{int(jr.P1)} / {int(jr.P2)} / {int(jr.P3)}")
    left, right = st.columns(2)
    fig = px.line(ims, x="D_tx_MB", y="information_eq22", markers=True, hover_data=["i_min"],
                  labels={"D_tx_MB": "D_tx (Mo)", "information_eq22": "information préservée"})
    fig.update_traces(line_color=GREEN)
    fig.add_trace(go.Scatter(x=[jr.D_tx_MB], y=[jr.information_eq22], mode="markers",
                             marker=dict(size=18, color="rgba(0,0,0,0)",
                                         line=dict(color=ORANGE, width=3))))
    fig.update_layout(title="Octets achetés par chaque seuil d'information", height=340,
                      showlegend=False)
    left.plotly_chart(fig, width="stretch")
    lv = ims.melt(id_vars="i_min", value_vars=["P1", "P2", "P3"], var_name="niveau",
                  value_name="objets")
    fig = px.bar(lv, x="i_min", y="objets", color="niveau",
                 color_discrete_map={"P1": GREEN, "P2": ORANGE, "P3": RED},
                 labels={"i_min": "I_min"})
    fig.update_layout(title="Niveaux choisis par le solveur", height=340)
    right.plotly_chart(fig, width="stretch")
    ws = res.joint[res.joint.sweep == "weights"]
    if ws.D_tx_MB.nunique() == 1:
        st.caption("Les 7 pondérations (α, β, γ) testées donnent la même solution : énergie, "
                   "octets et latence augmentent tous avec la taille d'un niveau, donc ce sont "
                   "les contraintes, pas les poids, qui décident.")


def page_scene(res) -> None:
    st.header("Scène — ce que voit le satellite et ce qu'il envoie")
    banner(res)
    ex = res.meta.get("example_scene")
    if not ex or not res.scene_image.exists():
        st.error("Pas de scène exemple : relancer scripts/paper_b0_b4.py.")
        return
    c = st.columns(3)
    show_gt = c[0].toggle("Vérité terrain", value=False)
    show_grid = c[1].toggle(f"Grille SAHI ({ex['tile']} px, {pct(ex['overlap'], 0)})", value=True)
    pad = c[2].slider("Marge des cadres (px)", 0, 30, 10)
    s = ex["scale"]
    img = np.asarray(Image.open(res.scene_image).convert("RGB"))
    fig = px.imshow(img)
    if show_grid:
        for x, y, w, h in slice_grid(ex["height"], ex["width"], ex["tile"], ex["overlap"]):
            fig.add_shape(type="rect", x0=x * s, y0=y * s, x1=(x + w) * s, y1=(y + h) * s,
                          line=dict(color="rgba(255,255,255,0.25)", width=1))
    det = res.scene[res.scene.kind == "detection"]
    for _, d in det.iterrows():
        fig.add_shape(type="rect", x0=(d.x - pad) * s, y0=(d.y - pad) * s,
                      x1=(d.x + d.w + pad) * s, y1=(d.y + d.h + pad) * s,
                      line=dict(color=PRIO_COLOR.get(int(d.priority), GREY), width=2))
    if show_gt:
        for _, g in res.scene[res.scene.kind == "ground_truth"].iterrows():
            fig.add_shape(type="rect", x0=g.x * s, y0=g.y * s, x1=(g.x + g.w) * s,
                          y1=(g.y + g.h) * s, line=dict(color="white", width=1, dash="dot"))
    fig.update_layout(height=720, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    left, right = st.columns([3, 1])
    left.plotly_chart(fig, width="stretch")
    right.markdown(f"**Scène {ex['width']} × {ex['height']} px**")
    for p, name in PRIORITY_FR.items():
        n = int((det.priority == p).sum())
        right.markdown(f"<span style='color:{PRIO_COLOR[p]}'>■</span> {name} : **{n}**",
                       unsafe_allow_html=True)
    right.markdown(f"Vrais navires : **{int((res.scene.kind == 'ground_truth').sum())}**")
    right.caption("Plus la confiance est basse, plus on joint d'images : P1 seul, P2 avec une "
                  "imagette, P3 avec une imagette et une vue de contexte (Tableau I du papier).")


def page_ablations(res) -> None:
    st.header("Ablations — que coûte chaque brique retirée ?")
    banner(res)
    d = ablation_deltas(res.ablations)
    left, right = st.columns(2)
    for col, title, where, good_up in (("rappel", "Écart de rappel (points)", left, True),
                                       ("octets (%)", "Écart d'octets envoyés", right, False)):
        colors = [GREEN if (v >= 0) == good_up or abs(v) < 1e-9 else RED for v in d[col]]
        fig = go.Figure(go.Bar(x=d[col], y=d.ablation, orientation="h", marker_color=colors))
        fig.update_layout(title=title, height=330, xaxis_tickformat=".0%" if "%" in col else ".2f")
        where.plotly_chart(fig, width="stretch")
    st.markdown("**Écart par rapport à B4 complet**")
    st.dataframe(d.style.format({"rappel": "{:+.3f}", "précision": "{:+.3f}",
                                 "octets (%)": "{:+.0%}", "énergie (%)": "{:+.0%}",
                                 "latence (%)": "{:+.0%}"}), hide_index=True, width="stretch")


def page_measured() -> None:
    st.header("Résultats mesurés — pipeline WP sur tuiles Airbus")
    m = measured()
    if m is None:
        st.error("Fichiers du pipeline WP absents de results/.")
        return
    st.success("**REAL** : 5 320 tuiles Airbus de test, détecteur YOLOv8n entraîné, orbite SGP4.")
    k = st.columns(4)
    best = m.strategies.iloc[-1]
    k[0].metric("Navires livrés", fr(best.ship_recall, 3), help="fourchette 0,40–0,82 selon le nuage")
    k[1].metric("Réduction", "×557", help="60 124 Mo offerts → 108 Mo envoyés par jour")
    k[2].metric("Détecteur mAP50", fr(m.detector["test"]["mAP50"], 3))
    k[3].metric("Gate : rappel", pct(m.gate["at_99_recall"]["ship_recall"]),
                help=f"{m.gate['params']} paramètres, {m.gate['infer_ms_per_tile']} ms/tuile")
    t = m.strategies[["strategy", "ship_recall", "ceiling_recall", "downlink_eff"]].copy()
    t["strategy"] = t.strategy.map(lambda s: STRATEGY_FR.get(s, s))
    t.columns = ["stratégie", "navires livrés", "plafond (détectés)", "efficacité liaison"]
    left, right = st.columns(2)
    left.markdown("**Stratégies (wp6_simulate_real.py)**")
    left.dataframe(t.round(3), hide_index=True, width="stretch")
    sizes = pd.DataFrame(m.detector["by_size"]).rename(columns={"bucket": "taille", "ships": "navires",
                                                                "recall": "rappel"})
    right.markdown("**Rappel du détecteur par taille (wp1)**")
    right.dataframe(sizes.round(3), hide_index=True, width="stretch")
    right.markdown("**Regret par politique (wp8) — plus petit = mieux**")
    right.dataframe(m.regret.rename(columns={"policy": "politique"}).round(4), hide_index=True,
                    width="stretch")
    cols = st.columns(2)
    for i, (title, path) in enumerate(m.figures.items()):
        cols[i % 2].image(str(path), caption=title)


@st.cache_data
def imported():
    return load_imported()


def page_imported() -> None:
    st.header("Mesures réelles — fusion avec oblhdj")
    m = imported()
    if m is None:
        st.error("results/oblhdj/ absent.")
        return
    st.success(f"**REAL** : YOLOv8n sur {m.swaths['n_swaths']} bandes de {m.swaths['swath_px']} px "
               f"faites de vraies tuiles Airbus ({m.swaths['n_ships']} navires, "
               f"{fr(m.swaths['pct_straddling_seam'], 1)} % coupés par un bord). Mesuré dans "
               "oblhdj/iastam-p7-semantic-downlink (commit d69f85d) ; voir results/oblhdj/PROVENANCE.md.")
    s = m.sahi.set_index("policy")
    k = st.columns(4)
    k[0].metric("SAHI 768 px : navires sur un bord", fr(s.loc["SAHI 768 px", "on_seam"], 3),
                delta=f"{fr(100 * (s.loc['SAHI 768 px', 'on_seam'] - s.loc['naive', 'on_seam']), 1)} pts "
                      "vs grille naïve")
    k[1].metric("SAHI 768 px : calcul", f"×{fr(s.loc['SAHI 768 px', 'compute_x'])}")
    k[2].metric("SAHI 512 px (papier) : calcul", f"×{fr(s.loc['SAHI 512 px', 'compute_x'])}",
                help=f"rappel global {fr(s.loc['SAHI 512 px', 'recall'], 3)}, contre "
                     f"{fr(s.loc['SAHI 768 px', 'recall'], 3)} à 768 px")
    k[3].metric("SAHI 512 px : petits navires", fr(s.loc["SAHI 512 px", "small"], 3),
                delta=f"{fr(100 * (s.loc['SAHI 512 px', 'small'] - s.loc['SAHI 768 px', 'small']), 1)} pts "
                      "vs 768 px")
    left, right = st.columns(2)
    t = m.sahi.melt(id_vars="policy", value_vars=["recall", "small", "on_seam"],
                    var_name="mesure", value_name="rappel")
    t["mesure"] = t.mesure.map({"recall": "tous", "small": "petits", "on_seam": "sur un bord"})
    fig = px.bar(t, x="policy", y="rappel", color="mesure", barmode="group",
                 color_discrete_sequence=[OCEAN, ORANGE, GREEN], labels={"policy": ""})
    fig.update_layout(title="Rappel par découpage (REAL)", yaxis_range=[0, 1.05], height=360,
                      legend_title="")
    left.plotly_chart(fig, width="stretch")
    fig = px.scatter(m.sahi, x="compute_x", y="recall", text="policy",
                     labels={"compute_x": "calcul relatif au découpage régulier", "recall": "rappel"})
    fig.update_traces(marker=dict(size=14, color=NAVY), textposition="top center")
    fig.update_layout(title="Ce que chaque découpage coûte et rapporte", height=360)
    right.plotly_chart(fig, width="stretch")
    st.warning("Le papier fixe des tuiles de 512 px. Sur ces bandes, 512 px coûte 4 fois le calcul "
               "du découpage régulier pour le même rappel global (+2,5 pts sur les petits navires "
               "seulement) ; 768 px coûte 1,56 fois. À écrire dans le papier.", icon="📏")
    lat = m.relay["latency_SIM_energy_free"]
    st.subheader("Relais inter-satellite (SIM, orbite SGP4, RAAN +90°)")
    k = st.columns(3)
    for col, (name, a, b) in zip(k, (("médiane", "direct_only_median_h", "with_relay_median_h"),
                                     ("90e centile", "direct_only_p90_h", "with_relay_p90_h"),
                                     ("maximum", "direct_only_max_h", "with_relay_max_h"))):
        col.metric(f"Latence {name}", f"{fr(lat[b])} h", delta=f"{fr(lat[b] - lat[a])} h vs direct",
                   delta_color="inverse")
    st.subheader("Énergie par tuile et par étape (temps REAL × puissances supposées, wp17)")
    st.dataframe(m.energy_stages.round(4), hide_index=True, width="stretch")


def goto(page: str | None = None, source: str | None = None) -> None:
    """Switch page and/or source on the next rerun (widgets cannot change once drawn)."""
    if page:
        st.session_state["goto_page"] = page
    if source:
        st.session_state["goto_source"] = source
    st.rerun()


def simulation_form() -> dict:
    """The parameters of paper_b0_b4.py, defaults equal to the script's own."""
    D = runner.DEFAULTS
    a, b, c = st.columns(3)
    a.markdown("**Scènes et détecteur**")
    source = a.selectbox("Source", runner.SOURCES, index=runner.SOURCES.index(D["source"]),
                         key="sim_source", format_func={"synthetic": "synthétique (SIM)",
                                                        "swaths": "24 bandes Airbus réelles",
                                                        "airbus": "mosaïques Airbus 2×2"}.get)
    detector = a.selectbox("Détecteur", ["classic", "yolo"], key="sim_detector",
                           format_func={"classic": "classique (secours, CPU)",
                                        "yolo": "YOLOv8n entraîné (GPU, .venv312)"}.get)
    scenes = a.slider("Nombre de scènes", 1, 48, D["scenes"], key="sim_scenes")
    scene_px = a.select_slider("Côté d'une scène synthétique (px)", [1024, 1536, 2048, 3072],
                               value=D["scene_px"], key="sim_scene_px",
                               disabled=source != "synthetic")
    seed = a.number_input("Graine", 0, 10_000, D["seed"], key="sim_seed")

    b.markdown("**SAHI et détection**")
    tile = b.select_slider("Tuile SAHI (px)", [256, 384, 512, 640, 768, 1024], value=D["tile"],
                           key="sim_tile")
    overlap = b.slider("Chevauchement", 0.0, 0.5, D["overlap"], 0.05, key="sim_overlap")
    fusion = b.radio("Fusion", ["nmm", "nms"], horizontal=True, key="sim_fusion",
                     format_func={"nmm": "NMM (fusion)", "nms": "NMS (suppression)"}.get)
    conf = b.slider("Seuil de confiance", 0.05, 0.70, D["conf"], 0.05, key="sim_conf")

    c.markdown("**Liaison**")
    link = c.selectbox("Liaison", list(LINK_PRESETS), key="sim_link")
    stations = c.multiselect("Stations sol", list(STATION_PRESETS), default=D["stations"].split(","),
                             key="sim_stations") or D["stations"].split(",")
    deadline = c.slider("Échéance (h)", 1.0, 24.0, D["deadline_hours"], 0.5, key="sim_deadline")
    with c.expander("Objectif joint (éq. 20-22)"):
        abg = [st.number_input(n, 0.0, 10.0, float(v), 0.5, key=f"sim_{k}")
               for n, k, v in zip(("α (énergie)", "β (octets)", "γ (latence)"),
                                  ("alpha", "beta", "gamma"), D["weights_abg"].split(","))]
        a_min = st.slider("A_min (éq. 21)", 0.5, 1.0, D["a_min"], 0.05, key="sim_a_min")
        i_min = st.slider("I_min (éq. 22)", 0.0, 1.0, D["i_min"], 0.05, key="sim_i_min")
    p = runner.normalise({
        "source": source, "detector": detector, "scenes": scenes, "scene_px": scene_px,
        "tile": tile, "overlap": round(overlap, 2), "fusion": fusion, "conf": round(conf, 2),
        "link": link, "stations": ",".join(stations), "deadline_hours": round(deadline, 1),
        "weights_abg": ",".join(f"{v:g}" for v in abg), "a_min": round(a_min, 2),
        "i_min": round(i_min, 2), "seed": seed})
    est = runner.estimate_seconds(p)
    a.caption(f"Durée estimée : **~{est:.0f} s** (CPU, mesurée sur le portable de dev)" if est
              else "Durée : dépend du GPU et des données, pas d'estimation.")
    return p


def run_live(p: dict) -> None:
    """Launch the script, tail its log into the page, then show the run or the failure."""
    job = runner.start(p)
    bar = st.progress(0.0, "Démarrage du script…")
    box = st.empty()
    done, msg, seen = 0.0, "Démarrage du script…", 0
    while True:
        running = job.proc.poll() is None
        lines = job.run.log.split("\n")
        complete_lines = lines if not running else lines[:-1]   # the last one may be partial
        for line in complete_lines[seen:]:
            done, new = runner.progress(line, done, p["scenes"])
            msg = new or msg
        seen = len(complete_lines)
        bar.progress(done, f"{msg} · {time.perf_counter() - job.t0:.0f} s")
        box.code("\n".join(ln for ln in lines if ln.strip())[-4000:] or "…", language=None)
        if not running:
            break
        time.sleep(0.4)
    run = runner.finish(job)
    if run.info["returncode"] == 0 and run.complete:
        st.session_state["sim_last"] = run.id
        goto(source=LIVE_PREFIX + run.id)
    bar.empty()
    st.error(f"**Le run a échoué** (code {run.info['returncode']}) : "
             f"{runner.failure_reason(run.log)}", icon="🛑")


def page_simulation() -> None:
    st.header("Simulation — lancer la chaîne du papier")
    st.caption("Lance `scripts/paper_b0_b4.py` avec ces paramètres dans `results/live/<run>/`. "
               "Les fichiers committés de `results/` ne sont jamais touchés, et chaque chiffre "
               "affiché ensuite vient des fichiers écrits par le script.")
    p = simulation_form()
    label = runner.expected_label(p)
    problems = runner.preflight(p)
    cached = runner.find_cached(p)
    st.markdown("---")
    left, right = st.columns([2, 3])
    (left.success if label == "REAL" else left.warning)(
        f"Ce run sera **{label}**" + ("" if label == "REAL" else
                                      " : il faut de vraies scènes ET le détecteur YOLO pour REAL."),
        icon="✅" if label == "REAL" else "⚠️")
    for msg in problems:
        right.error(msg, icon="🛑")
    if cached and not problems:
        secs = (cached.info or {}).get("seconds")
        right.info(f"Ces paramètres ont déjà été calculés : run `{cached.id}`"
                   + (f" ({secs:.0f} s)" if secs else "") + ".", icon="♻️")
        c1, c2 = right.columns(2)
        if c1.button("Charger ce run", type="primary", key="sim_load"):
            st.session_state["sim_last"] = cached.id
            goto(source=LIVE_PREFIX + cached.id)
        relaunch = c2.button("Relancer quand même", key="sim_rerun")
    else:
        relaunch = right.button("Lancer", type="primary", disabled=bool(problems), key="sim_go")
    with st.expander("Ligne de commande"):
        st.code(" ".join(f'"{a}"' if (" " in a or not a) else a
                         for a in runner.command(p, runner.LIVE / "<run>")), language="bash")
    if relaunch:
        run_live(p)
        return

    last = st.session_state.get("sim_last")
    if last and (runner.LIVE / last / "paper_b0_b4.json").exists():
        res = paper(LIVE_PREFIX + last)
        st.subheader(f"Dernier run : {source_name(LIVE_PREFIX + last)}")
        banner(res)
        c = res.campaign.set_index("config")
        k = st.columns(4)
        k[0].metric("Réduction B4", f"×{fr(c.loc['B4'].reduction_x, 0)}")
        k[1].metric("Rappel B4", fr(c.loc["B4"].recall))
        k[2].metric("Énergie économisée", pct(c.loc["B4"].ES))
        k[3].metric("Latence médiane B4", f"{fr(c.loc['B4'].T_total_med_h)} h")
        b = st.columns(3)
        if b[0].button("Voir la vue d'ensemble", key="sim_view"):
            goto(page="Vue d'ensemble", source=LIVE_PREFIX + last)
        if b[1].button("Voir la communication", key="sim_comms"):
            goto(page="Communication", source=LIVE_PREFIX + last)
        if b[2].button("Comparer à la référence", key="sim_cmp"):
            st.session_state["cmp_b_goto"] = LIVE_PREFIX + last
            goto(page="Comparer")

    runs = runner.live_runs()
    if runs:
        st.subheader("Runs précédents")
        st.dataframe(pd.DataFrame([{
            "run": r.id, "label": r.label, "source": r.params["source"],
            "détecteur": r.params["detector"], "scènes": r.params["scenes"],
            "tuile": r.params["tile"], "seuil": r.params["conf"],
            "durée (s)": (r.info or {}).get("seconds"),
            "état": "terminé" if r.complete else
                    ("échec" if r.info else "en cours ou interrompu")} for r in runs]),
            hide_index=True, width="stretch")


def page_compare() -> None:
    st.header("Comparer — deux runs côte à côte")
    srcs = available_sources() + live_sources()
    if len(srcs) < 2:
        st.info("Il faut au moins deux runs : en lancer un depuis la page « Simulation ».")
        return
    goto_b = st.session_state.pop("cmp_b_goto", None)
    if goto_b in srcs:
        st.session_state["cmp_b"] = goto_b
        st.session_state["cmp_a"] = "SIM" if "SIM" in srcs and goto_b != "SIM" else srcs[0]
    if st.session_state.get("cmp_a") not in srcs:
        st.session_state["cmp_a"] = "SIM" if "SIM" in srcs else srcs[0]
    if st.session_state.get("cmp_b") not in srcs:
        st.session_state["cmp_b"] = next(s for s in srcs if s != st.session_state["cmp_a"])
    c = st.columns(2)
    a = c[0].selectbox("Run A (référence)", srcs, key="cmp_a", format_func=source_name)
    b = c[1].selectbox("Run B", srcs, key="cmp_b", format_func=source_name)
    ra, rb = paper(a), paper(b)
    if ra.label != rb.label:
        st.warning(f"A est **{ra.label}**, B est **{rb.label}** : l'écart mélange un changement de "
                   "paramètres et un changement de données ou de détecteur.", icon="⚠️")
    else:
        st.caption(f"A et B sont tous deux **{ra.label}**.")
    metrics = st.multiselect("Mesures", list(COMPARE_METRICS), key="cmp_metrics",
                             default=["D_tx_MB", "recall", "F1", "IP", "E_total_J", "T_total_med_h"],
                             format_func=COMPARE_METRICS.get)
    if metrics:
        t = compare_campaigns(ra.campaign, rb.campaign, metrics)
        delta = [col for col in t.columns if col.endswith("· Δ")]
        st.dataframe(t.style.format("{:.4g}", subset=[col for col in t.columns[1:] if col not in delta])
                     .format("{:+.4g}", subset=delta), hide_index=True, width="stretch")
        m = st.selectbox("Graphique", metrics, format_func=COMPARE_METRICS.get, key="cmp_chart")
        both = pd.concat([ra.campaign.assign(run="A"), rb.campaign.assign(run="B")])
        fig = px.bar(both, x="config", y=m, color="run", barmode="group",
                     color_discrete_map={"A": GREY, "B": OCEAN},
                     log_y=m in ("D_tx_MB", "reduction_x", "E_total_J"),
                     labels={"config": "", m: COMPARE_METRICS[m]})
        fig.update_layout(title=f"{COMPARE_METRICS[m]} : A contre B", height=360, legend_title="")
        st.plotly_chart(fig, width="stretch")
    sa, sb = run_settings(ra.meta), run_settings(rb.meta)
    s = pd.DataFrame({"A": sa, "B": sb}).astype(str)
    s["diffère"] = ["●" if x != y else "" for x, y in zip(s.A, s.B)]
    st.markdown("**Paramètres des deux runs** (lus dans leur `paper_b0_b4.json`)")
    st.dataframe(s, width="stretch")


def page_assumptions(res) -> None:
    st.header("Hypothèses — ce qui n'est pas mesuré")
    banner(res)
    meta = res.meta
    st.markdown(f"- **Source** : {meta['source']} · **détecteur** : {meta['detector']} · "
                f"**scènes** : {meta['scenes']}\n"
                f"- **Liaison** : {meta['link']} · **échéance** : {meta['deadline_h']} h")
    left, right = st.columns(2)
    left.markdown("**Puissances supposées (W)**")
    left.dataframe(pd.Series(meta["powers_W_ASSUMED"], name="W"), width="stretch")
    right.markdown("**Relais inter-satellite (supposé)**")
    right.dataframe(pd.Series(meta["relay_ASSUMED"], name="valeur").astype(str), width="stretch")
    right.markdown("**Configuration SAHI**")
    right.dataframe(pd.Series(meta["sahi"], name="valeur").astype(str), width="stretch")
    left.markdown("**Fenêtres de liaison sur l'horizon**")
    left.dataframe(pd.Series(meta["links"], name="nombre"), width="stretch")
    if "stations" in meta:
        left.markdown(f"**Stations sol** : {', '.join(meta['stations'])} · débit moyen "
                      f"{fr(meta['mean_downlink_Mbps'])} Mbit/s")
    dt = meta.get("det_timing", {"mode": "measured"})
    right.markdown("**Temps du détecteur pour E_det**")
    right.write(f"{dt['mode']}" + (f" — {fr(dt['ms_per_inference'])} ms/inférence ({dt['source']})"
                                   if "ms_per_inference" in dt else " — temps mesuré pendant ce run"))
    if "joint_eq20" in meta:
        j = meta["joint_eq20"]
        right.markdown("**Éq. 20 (supposé)**")
        right.dataframe(pd.Series({**{f"w_{k}": v for k, v in j["weights"].items()},
                                   **j["constraints"], "seuil candidats": j["candidate_floor"]},
                                  name="valeur").astype(str), width="stretch")
    st.caption("Mesuré : détections, octets de chaque paquet, temps de chaque étape, passages et "
               "géométrie ISL (SGP4). Supposé : tout ce qui est listé sur cette page.")


# ------------------------------------------------------------------ app
def main() -> None:
    st.set_page_config(page_title="SAT7 · Downlink", page_icon="🛰️", layout="wide")
    st.sidebar.title("🛰️ SAT7 · Downlink")
    st.sidebar.caption("Tableau de bord — station sol · lit code/results/, lance des runs depuis "
                       "« Simulation »")
    standalone = {"Simulation": page_simulation, "Comparer": page_compare,       # need no
                  "Résultats mesurés (WP)": page_measured,                       # selected run
                  "Mesures réelles (fusion)": page_imported}
    for k in ("page", "source"):                       # set by goto() before this rerun
        if (v := st.session_state.pop(f"goto_{k}", None)) is not None:
            st.session_state[k] = v
    asked = st.query_params.get("page")                # ?page=Communication opens that page
    if "page" not in st.session_state and asked in PAGES:
        st.session_state["page"] = asked
    page = st.sidebar.radio("Page", PAGES, key="page")
    live = live_sources()
    sources = available_sources() + live[:LIVE_IN_SIDEBAR]
    chosen = st.session_state.get("source")
    if chosen in live and chosen not in sources:       # an older live run picked elsewhere
        sources.append(chosen)
    if not sources and page not in standalone:
        st.error("Aucun résultat du papier : lancer une simulation depuis la page « Simulation » "
                 "ou `scripts/paper_b0_b4.py`.")
        return
    if chosen is not None and chosen not in sources:
        del st.session_state["source"]                 # its files were removed
    source = st.sidebar.radio("Source", sources, key="source", format_func=source_name,
                              help="REAL = YOLO sur vraies scènes ; SIM = synthétique ; "
                                   "LIVE = lancé depuis « Simulation »") if sources else None
    if st.sidebar.button("Recharger les fichiers"):
        st.cache_data.clear()
    st.sidebar.markdown("---")
    st.sidebar.caption("Pour de nouveaux chiffres : la page « Simulation », ou relancer "
                       "`scripts/paper_b0_b4.py` puis « Recharger les fichiers ».")
    if page in standalone:
        standalone[page]()
        return
    res = paper(source)
    {"Vue d'ensemble": page_overview, "Communication": page_comms, "Compromis": page_tradeoff,
     "Scène": page_scene, "Ablations": page_ablations, "Hypothèses": page_assumptions}[page](res)


main()
