"""Build the French pipeline guide (PDF): every file, how they connect, and the dashboard advice.

    .venv/Scripts/python.exe scripts/guide_figures.py     # figures first (needs skyfield)
    python scripts/build_pipeline_guide.py                # then this (needs reportlab)

Output: ../paper/IASTAM_P7_Guide_Pipeline_FR.pdf. Numbers are read from results/paper_*_sim.csv.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RES = ROOT / "results"
FIG = RES / "guide"

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Arial", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", str(FONTS / "arialbi.ttf")))
pdfmetrics.registerFont(TTFont("Consolas", str(FONTS / "consola.ttf")))
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                   boldItalic="Arial-BoldItalic")

NAVY, OCEAN, ORANGE = colors.HexColor("#0B2545"), colors.HexColor("#1B6CA8"), colors.HexColor("#E07A1F")
GREEN, RED, MUTED = colors.HexColor("#2E8B57"), colors.HexColor("#C0392B"), colors.HexColor("#6B7280")
PAPER = colors.HexColor("#F3F6FA")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Arial", fontSize=9.6, leading=13.6,
                      spaceAfter=5, alignment=TA_LEFT)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.5, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.1, leading=10.3, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Arial-Bold", textColor=colors.white)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Arial-Bold", fontSize=16, leading=20,
                    textColor=NAVY, spaceBefore=4, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Arial-Bold", fontSize=12, leading=15,
                    textColor=OCEAN, spaceBefore=10, spaceAfter=5)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
CALLOUT = ParagraphStyle("callout", parent=BODY, backColor=colors.HexColor("#FDF1E6"),
                         borderColor=ORANGE, borderWidth=0.8, borderPadding=6, leftIndent=6,
                         rightIndent=6, spaceBefore=6, spaceAfter=10)
GOODBOX = ParagraphStyle("good", parent=CALLOUT, backColor=colors.HexColor("#EAF6EF"), borderColor=GREEN)
CAPTION = ParagraphStyle("caption", parent=SMALL, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
CODE = ParagraphStyle("code", parent=BODY, fontName="Consolas", fontSize=7.4, leading=9.2,
                      backColor=colors.HexColor("#F5F7FA"), borderColor=colors.HexColor("#DDE3EA"),
                      borderWidth=0.5, borderPadding=5, leftIndent=4, rightIndent=4, spaceAfter=10)
TITLE = ParagraphStyle("title", parent=H1, fontSize=24, leading=29, alignment=TA_CENTER)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=12, leading=16, alignment=TA_CENTER, textColor=OCEAN)

W = A4[0] - 3.6 * cm


def P(t, s=BODY):
    return Paragraph(t, s)


def C(t):
    return f'<font name="Consolas" size="8.3">{t}</font>'


def bullets(items):
    return [Paragraph(i, BULLET, bulletText="•") for i in items]


def fig(name, width=W, caption=None):
    path = FIG / name if (FIG / name).exists() else RES / name
    iw, ih = ImageReader(str(path)).getSize()
    out = [Image(str(path), width=width, height=width * ih / iw)]
    if caption:
        out.append(P(caption, CAPTION))
    return KeepTogether(out)


def table(rows, widths, head=OCEAN, zebra=True):
    data = [[P(c, CELLB) for c in rows[0]]] + [[P(str(c), CELL) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), head), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDE3EA")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        st += [("BACKGROUND", (0, i), (-1, i), PAPER) for i in range(2, len(data), 2)]
    t.setStyle(TableStyle(st))
    return t


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.8 * cm, 1.1 * cm, "IASTAM P7 · ENIS — Guide du pipeline (FR)")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"page {doc.page}")
    canvas.restoreState()


def fr(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


def build():
    tab = pd.read_csv(RES / "paper_b0_b4_sim.csv").set_index("config")
    abl = pd.read_csv(RES / "paper_ablations_sim.csv").set_index("config")
    sw = pd.read_csv(RES / "paper_tradeoff_sim.csv")
    b4 = tab.loc["B4"]
    s = []

    # ------------------------------------------------------------ cover
    s += [Spacer(1, 3.2 * cm), P("Guide du pipeline", TITLE),
          P("Energy-Aware Semantic Downlink for Maritime Object Detection<br/>"
            "with Optional Inter-Satellite Relay", SUB), Spacer(1, 0.6 * cm),
          P("Chaque fichier, ce qu'il fait, comment les fichiers sont reliés,<br/>"
            "et faut-il ajouter un tableau de bord ?", SUB), Spacer(1, 1.2 * cm),
          fig("pipeline_flow.png", W, "Le pipeline du papier tel qu'il est implémenté : en haut, "
              "à bord du satellite ; en bas, communication et station sol. Sous chaque étape : "
              "le fichier et la fonction qui la réalisent."),
          Spacer(1, 0.6 * cm),
          P(f"ENIS · IASTAM 6.0 · Track 4, Problème 7 — {date.today():%d/%m/%Y}", SUB),
          PageBreak()]

    # ------------------------------------------------------------ 1. reading guide
    s += [P("1. En une page", H1),
          P("Le dépôt contient <b>deux pipelines</b> qui partagent la même bibliothèque "
            f"{C('code/sat7/')} :"),
          *bullets([
              "<b>Le pipeline du papier</b> (nouveau, objet de ce guide) : grande image → tuiles "
              "512 px avec 20 % de chevauchement → SAHI + YOLO → coordonnées globales → fusion → "
              "enregistrements sémantiques P0–P3 → liaison directe ou relais inter-satellite → "
              f"station sol. Il est lancé par {C('scripts/paper_b0_b4.py')}, qui exécute la "
              "campagne B0 → B4 du papier, ses 6 ablations et ses courbes de compromis.",
              "<b>Le pipeline historique « WP »</b> (wp0 à wp14) : il a produit les résultats "
              "<b>mesurés sur les vraies images Airbus</b> (557×, rappel 0,685, etc.). Il reste "
              "la référence pour les chiffres réels ; il est résumé en section 7."]),
          P("<b>Règle de lecture.</b> Chaque équation du papier correspond à une fonction, et chaque "
            f"fonction est vérifiée par un test dans {C('tests/test_paper.py')}. Les résultats "
            "présentés ici sont étiquetés <b>SIM</b> : ils proviennent de scènes synthétiques et "
            "d'un détecteur classique de secours, car le dataset, les poids YOLO et l'environnement "
            "GPU ne sont pas sur cette machine. Ils montrent que la chaîne fonctionne ; ce ne sont "
            "pas des résultats YOLO.", CALLOUT),
          P("Ce qui a été nettoyé", H2),
          *bullets([
              f"Toute la chaîne embarquée est maintenant <b>un seul objet</b>, {C('OnboardPipeline')} "
              f"({C('sat7/pipeline.py')}). Ses étapes portent les noms des termes d'énergie de "
              f"l'éq. 9 : {C('pre')}, {C('tile')}, {C('det')}, {C('fusion')}, {C('semantic')}.",
              f"Les configurations B0–B4 et les six ablations sont de simples valeurs de "
              f"{C('PipelineConfig')} déclarées dans {C('experiment_plan()')} : il n'y a plus de "
              "variantes codées à la main dans le script.",
              f"Le détecteur ne tourne qu'<b>une fois par découpage</b> ({C('detection_key')}) ; "
              "les configurations qui ne diffèrent que par la fusion, le seuil ou la politique "
              "réutilisent ses détections.",
              f"Le chemin mort {C('raw=')} de {C('sahi_detect')} a été supprimé ; le côté sol "
              f"({C('ground_decode')}) et les centres livrés ({C('record_centres')}) sont dans "
              "le même module que le côté bord.",
              f"Un journal par paquet ({C('results/paper_packets_sim.csv')}) est écrit : c'est la "
              "source de données du tableau de bord.",
              "Résultat identique avant et après nettoyage (mêmes rappels, octets, IP) ; "
              "140 tests passent, 2 sont ignorés (ils demandent torch).",
              f"Le pipeline WP est archivé dans {C('code/legacy_wp/')} (modules, scripts, tests), "
              f"toujours exécutable ; {C('code/sat7/')} ne contient plus que le papier et le socle "
              f"partagé ; les journaux d'exécution sont dans {C('results/logs/')}."]),
          PageBreak()]

    # ------------------------------------------------------------ 2. step by step
    s += [P("2. Le pipeline du papier, étape par étape", H1),
          P("Chaque ligne suit une donnée de gauche à droite : ce qui entre dans l'étape, ce qui en "
            "sort, et où c'est codé."),
          table([["Étape (papier)", "Équation", "Fichier · fonction", "Entrée → sortie"],
                 ["Grande image", "éq. 1", "scenes.py · synthetic_scene, airbus_mosaic",
                  "dataset → image H×W×3 + vérité terrain"],
                 ["Prétraitement", "§III-B · E_pre", "pipeline.py · preprocess",
                  "image brute → tableau uint8 contigu + T_pre"],
                 ["Découpage en tuiles", "éq. 2 · E_tile", "sahi.py · slice_grid",
                  "H, W → liste (x0, y0, 512, 512), pas = 512×(1−0,2)"],
                 ["Détection par tuile (SAHI)", "éq. 3-4 · E_det", "sahi.py · sahi_detect ; "
                  "detectors.py · YoloDetector", "tuiles → d = (x, y, w, h, p, c) locales"],
                 ["Coordonnées globales", "éq. 5", "sahi.py · Detection.to_global",
                  "x_g = x0 + x_l, y_g = y0 + y_l"],
                 ["Fusion des doublons", "éq. 6 · E_fusion", "sahi.py · fuse (NMS ou NMM, IoU ou IoS)",
                  "détections de tuiles voisines → D_global"],
                 ["Enregistrement sémantique", "éq. 7", "semantic.py · SemanticRecord",
                  "détection → {classe, confiance, bbox, horodatage, id image}"],
                 ["Priorités P0–P3", "Tableau I", "semantic.py · PriorityPolicy, build_packet",
                  "confiance → rejet / métadonnées / + ROI / + contexte"],
                 ["Paquet transmis", "D_tx", "semantic.py · SemanticPacket.encode",
                  "enregistrements + JPEG → octets réels"],
                 ["Choix du chemin", "éq. 16, 25, 26", "relay.py · Network.send, PathPolicy",
                  "paquet prêt à t → direct ou relais (min J, échéance)"],
                 ["Liaison directe", "éq. 12-14, 23", "relay.py · LinkSchedule ; orbit.py",
                  "passages SGP4 au-dessus de Sfax → E = P_tx·D/R"],
                 ["Relais inter-satellite", "éq. 15, 24, 29", "relay.py · isl_windows, relay_orbit",
                  "fenêtres ISL + passages du relais → E_ISL + E_GS"],
                 ["Décodage au sol", "éq. 28", "pipeline.py · ground_decode",
                  "octets → enregistrements + T_decoding"],
                 ["Évaluation", "éq. 8-22, 27-30", "metrics.py ; energy.py",
                  "rappel, F1, mAP, IP, E_total, ES, DR, T_total, objectif J"]],
                [3.1 * cm, 2.2 * cm, 5.6 * cm, W - 10.9 * cm]),
          Spacer(1, 6),
          P("Les quatre objets qui circulent", H2),
          *bullets([
              f"{C('Detection')} (sahi.py) : une boîte d = (x, y, w, h, p, c). Tous les modules la "
              "partagent, c'est pourquoi sahi.py est au centre du graphe de dépendances.",
              f"{C('SahiResult')} : la sortie du détecteur sur une image, avec les détections brutes "
              "et les temps T_tile, T_det. Réutilisable par plusieurs configurations.",
              f"{C('OnboardOutput')} (pipeline.py) : les détections finales, le <b>paquet en octets</b> "
              "et le temps de chaque étape. C'est tout ce qui quitte le satellite.",
              f"{C('PathChoice')} (relay.py) : le chemin pris (direct, relais ou aucun), l'heure de "
              "livraison, E_comm et T_comm, plus les options écartées (pour l'éq. 35)."]),
          PageBreak()]

    # ------------------------------------------------------------ 3. files
    s += [P("3. Chaque fichier du pipeline du papier", H1),
          table([["Fichier", "Rôle", "Contenu principal", "Utilisé par"],
                 ["sat7/sahi.py", "Découpage, SAHI, coordonnées globales, fusion (éq. 1-6)",
                  "Detection, SahiConfig (tile=512, overlap=0,20, fusion, seuil), slice_grid, fuse, "
                  "sahi_detect, direct_detect", "pipeline, detectors, semantic, metrics, scenes"],
                 ["sat7/detectors.py", "Brancher un détecteur dans SAHI",
                  "YoloDetector (poids WP1, Ultralytics) ; ClassicDetector (secours sans torch, "
                  "pré-filtre classique + confiance par contraste)", "paper_b0_b4, guide_figures"],
                 ["sat7/semantic.py", "Représentation sémantique et politique (éq. 7, Tab. I)",
                  "SemanticRecord (24 octets, +8 si géo), PriorityPolicy (t_discard 0,25 / t_context "
                  "0,40 / t_meta 0,70), RoiConfig, build_packet, SemanticPacket.encode/decode",
                  "pipeline"],
                 ["sat7/pipeline.py", "Orchestration de bord + décodage sol",
                  "PipelineConfig, OnboardPipeline (detect, detections, run, with_threshold), "
                  "OnboardOutput, preprocess, ground_decode, record_centres", "paper_b0_b4, tests"],
                 ["sat7/energy.py", "Modèle énergie / latence / objectif (éq. 8-22, 27-30)",
                  "PowerConfig (HYPOTHÈSES), processing_energy, comm_energy_direct/relay, "
                  "energy_saving, downlink_reduction, total_latency, select_operating_point",
                  "relay, paper_b0_b4"],
                 ["sat7/relay.py", "Liaisons et choix du chemin (éq. 15-16, 23-26, 29)",
                  "Window, LinkSchedule (FIFO sur fenêtres), ground_windows, isl_windows, "
                  "RelayConfig (HYPOTHÈSES), PathPolicy, Network.send", "paper_b0_b4, guide_figures"],
                 ["sat7/metrics.py", "Métriques §VIII-E",
                  "match (IoU ≥ 0,5), DetAccumulator (précision, rappel, F1, AP, rappel des petits "
                  "objets &lt; 32 px), preserved (préservation de l'information)", "paper_b0_b4"],
                 ["sat7/scenes.py", "Grandes scènes (éq. 1)",
                  "airbus_mosaic (2×2 vraies tuiles Airbus = 1536 px, RÉEL) ; synthetic_scene "
                  "(SIM)", "paper_b0_b4, guide_figures"],
                 ["scripts/paper_b0_b4.py", "La campagne du papier (§VIII)",
                  "experiment_plan (B0-B4 + ablations), boucle de bord, transmit (réseau), summarise, "
                  "balayage du seuil, objectif éq. 20, sorties CSV/JSON/PNG", "—"],
                 ["scripts/guide_figures.py", "Figures de ce guide",
                  "schémas, graphe des fichiers, maquettes du tableau de bord", "build_pipeline_guide"],
                 ["scripts/build_pipeline_guide.py", "Ce PDF", "mise en page reportlab", "—"],
                 ["tests/test_paper.py", "Une vérification par équation",
                  "16 tests : tuiles, éq. 5, fusion, aller-retour du paquet, Tableau I, énergie, "
                  "objectif, liaisons, relais, métriques, chaîne complète", "pytest"]],
                [3.3 * cm, 3.6 * cm, 6.4 * cm, W - 13.3 * cm]),
          Spacer(1, 6),
          P("Fichiers existants réutilisés tels quels", H2),
          *bullets([
              f"{C('sat7/orbit.py')} : orbite SGP4, station Sfax, capacité de chaque passage "
              "(Shannon). Utilisé par relay.py pour les passages du satellite principal et du relais.",
              f"{C('sat7/prefilter.py')} : filtre classique (Otsu, morphologie). Utilisé par le "
              "détecteur de secours.",
              f"{C('sat7/synthetic.py')}, {C('sat7/rle.py')} : scènes synthétiques et boîtes."]),
          PageBreak()]

    # ------------------------------------------------------------ 4. relations
    s += [P("4. Les relations entre les fichiers", H1),
          fig("relations_paper.png", W, "Graphe des dépendances du pipeline du papier. Une flèche "
              "signifie « importe et utilise ». Il est extrait des imports réels du code."),
          P("Comment lire ce graphe", H2),
          *bullets([
              f"<b>Un seul point d'entrée</b> : {C('paper_b0_b4.py')} construit les scènes, choisit "
              "le détecteur, crée un OnboardPipeline par configuration, puis envoie les paquets dans "
              "le réseau et calcule les métriques.",
              "<b>pipeline.py ne dépend que de sahi.py et semantic.py</b> : la partie à bord ne sait "
              "rien de l'orbite ni de l'énergie. C'est voulu : c'est le code qui irait sur le "
              "satellite.",
              "<b>La communication est séparée</b> : relay.py (avec orbit.py et energy.py) ne reçoit "
              "que des tailles de paquets et des heures. Le relais ne voit jamais l'image, comme "
              "l'exige le papier.",
              "<b>Les sorties sont des fichiers</b> dans code/results/ : le PDF et le futur tableau "
              "de bord les <i>lisent</i>, ils ne relancent jamais le calcul."]),
          P("Le déroulement d'une exécution", H2),
          table([["#", "Ce qui se passe", "Où"],
                 ["1", "Pour chaque scène : prétraitement, puis le détecteur une fois par découpage "
                  "(512/20 %, 512/0 %, image entière)", "paper_b0_b4.main → OnboardPipeline.detect"],
                 ["2", "Pour chaque configuration : seuil + fusion + paquet, mesure de chaque temps",
                  "OnboardPipeline.run"],
                 ["3", "Décodage du paquet comme au sol, centres des enregistrements livrés",
                  "ground_decode, record_centres"],
                 ["4", "Fenêtres de liaison sur 48 h : passages principal, ISL, passages relais",
                  "ground_windows, isl_windows"],
                 ["5", "Paquets envoyés dans l'ordre de capture, chemin choisi par min J sous échéance",
                  "transmit → Network.send"],
                 ["6", "Métriques, énergie, objectif, CSV + JSON + figure", "summarise, select_operating_point"]],
                [0.7 * cm, 9.5 * cm, W - 10.2 * cm]),
          PageBreak()]

    # ------------------------------------------------------------ 5. campaign
    s += [P("5. Campagne B0 → B4 et ablations", H1),
          P("Chaque configuration du Tableau II du papier est une valeur de PipelineConfig ; "
            "les ablations du §VIII-G changent un seul champ par rapport à B4."),
          table([["Config.", "Ce qui est fait à bord", "Ce qui est envoyé", "PipelineConfig"],
                 ["B0", "rien", "image brute", "— (octets bruts)"],
                 ["B1", "YOLO sur l'image entière", "sortie de détection", "sahi=False, semantic=False"],
                 ["B2", "SAHI + YOLO", "sortie de détection", "semantic=False"],
                 ["B3", "SAHI + YOLO + politique", "métadonnées / ROI, direct", "défaut, relais interdit"],
                 ["B4", "SAHI + YOLO + politique", "direct ou relais", "défaut, relais permis"]],
                [1.6 * cm, 4.4 * cm, 4.4 * cm, W - 10.4 * cm]),
          Spacer(1, 8),
          P('Résultats <font color="#1B6CA8">[SIM]</font> — 24 scènes de 2048 × 2048', H2),
          table([["Config.", "D_tx (Mo)", "Réduction", "Rappel", "F1", "IP", "E_total (J)", "ES",
                  "Latence méd."]] +
                [[k, f"{r.D_tx_MB:.3g}", f"×{r.reduction_x:,.0f}".replace(",", " "), fr(r.recall),
                  fr(r.F1), fr(r.IP), f"{r.E_total_J:,.0f}".replace(",", " "), f"{r.ES:.1%}".replace(".", ","),
                  f"{fr(r.T_total_med_h)} h"] for k, r in tab.iterrows()],
                [1.5 * cm] + [(W - 1.5 * cm) / 8] * 8),
          Spacer(1, 8),
          *bullets([
              f"<b>SAHI compte</b> : rappel {fr(tab.loc['B1'].recall)} sans découpage contre "
              f"{fr(tab.loc['B2'].recall)} avec, pour environ "
              f"{tab.loc['B2'].E_proc_J / tab.loc['B1'].E_proc_J:.0f}× plus d'énergie de calcul.",
              f"<b>Le relais divise la latence</b> : {fr(tab.loc['B3'].T_total_med_h)} h en direct "
              f"seul (B3) contre {fr(b4.T_total_med_h)} h (B4) ; {b4.relay_share:.0%} des paquets "
              "passent par l'ISL.",
              f"<b>Ablations</b> : sans fusion, la précision tombe de {fr(abl.loc['B4 full'].precision)} "
              f"à {fr(abl.loc['without fusion'].precision)} (doublons) ; sans politique adaptative, "
              f"{abl.loc['without adaptive downlink'].D_tx_MB / abl.loc['B4 full'].D_tx_MB:.1f}× plus "
              "d'octets.",
              f"<b>Objectif éq. 20</b> (F1 ≥ 0,5, IP ≥ 0,9) : meilleur seuil "
              f"{fr(sw.loc[(sw.det_thr - 0.55).abs().idxmin(), 'det_thr'])}."]),
          fig("paper_tradeoffs_sim.png", W * 0.86, "Courbes de compromis du papier (éq. 32 à 35), "
              "produites par paper_b0_b4.py [SIM]."),
          PageBreak()]

    # ------------------------------------------------------------ 6. measured vs assumed + run
    s += [P("6. Mesuré, supposé, et comment l'exécuter", H1),
          table([["Mesuré", "Supposé (affiché à chaque exécution)"],
                 ["détections ; octets de chaque paquet (sérialisation binaire réelle et JPEG réels) ; "
                  "temps de chaque étape T_k sur la machine ; passages au-dessus de Sfax et géométrie "
                  "ISL (SGP4)",
                  "puissances P_k (PowerConfig) ; débits des liaisons et de l'ISL ; orbite du relais "
                  "(RAAN +60°) ; heures de capture ; échéance 6 h ; poids λ_E, λ_T ; ciel des scènes "
                  "synthétiques"]],
                [W / 2, W / 2], zebra=False),
          Spacer(1, 6),
          *bullets([
              "Les temps viennent d'un ordinateur portable, pas du matériel de vol : l'énergie est une "
              "<b>estimation</b>.",
              "Le papier cite xView, VisDrone, DOTA et HRSID ; le code utilise <b>Airbus</b> (optique), "
              "le seul jeu de données préparé. Les grandes scènes réelles sont des mosaïques 2×2 : "
              "les jointures entre tuiles sont artificielles."]),
          P("Commandes", H2),
          Preformatted("cd code\n"
                       "# tests (140 réussis, 2 ignorés sans torch)\n"
                       ".venv/Scripts/python.exe -m pytest tests legacy_wp/tests -q\n\n"
                       "# campagne SIM, n'importe où\n"
                       ".venv/Scripts/python.exe scripts/paper_b0_b4.py\n\n"
                       "# campagne RÉELLE : machine GPU avec data/ et runs/\n"
                       ".venv312/Scripts/python.exe scripts/paper_b0_b4.py --source airbus \\\n"
                       "    --detector yolo --scenes 200\n\n"
                       "# ce guide\n"
                       ".venv/Scripts/python.exe scripts/guide_figures.py\n"
                       "python scripts/build_pipeline_guide.py", CODE),
          P("Tous les paramètres du §VIII-D sont des options : --tile, --overlap, --fusion, --conf, "
            "--link, --deadline-hours, plus PowerConfig et RelayConfig dans le code."),
          PageBreak()]

    # ------------------------------------------------------------ 7. WP pipeline
    s += [P("7. Le pipeline historique « WP » (résultats réels)", H1),
          P(f"<b>Emplacement :</b> {C('code/legacy_wp/')} (archivé, toujours exécutable), sauf wp0, "
            f"wp1 et wp2, restés dans {C('code/scripts/')} car le pipeline du papier en a besoin.",
            SMALL),
          P("Avant l'implémentation du papier, le projet a été construit en lots de travail (WP). "
            "Ces scripts ont produit tous les chiffres <b>mesurés sur Airbus</b> du README et des "
            "rapports. Ils partagent sat7/ avec le pipeline du papier mais ne l'appellent pas."),
          fig("relations_wp.png", W, "Les scripts WP, de gauche à droite dans l'ordre où leurs "
              "sorties s'enchaînent."),
          table([["Script", "Ce qu'il fait"],
                 ["wp0_build_dataset", "dataset Airbus sans fuite (doublons regroupés, sat7/dedup)"],
                 ["wp1_train_detector · wp1_export_int8", "entraîne YOLOv8n (mAP50 0,804) ; export ONNX FP32/INT8"],
                 ["wp2_predict_split · wp2_calibrate", "prédictions sur val ; calibration de la confiance"],
                 ["eval_prefilter · wp3_train_gate · wp3_score_tiles",
                  "filtre classique (échec : 0,645) ; petit CNN « gate » (0,988 en 0,15 ms)"],
                 ["wp4_measure_lod · wp6_fit_size_model · wp7_measure_cheap_products",
                  "taille réelle de chaque niveau de détail ; loi octets = a·taille^b"],
                 ["contact_windows", "passages au-dessus de Sfax et capacité"],
                 ["wp6_build_catalogue · wp6_simulate_real",
                  "catalogue des tuiles réelles ; résultat principal 557×, rappel 0,685"],
                 ["simulate_day", "première version sur journée synthétique"],
                 ["wp7_budget_sweep · wp8_queue_aware · wp14_gate_signal",
                  "où vont les octets ; adaptation à la file ; gate appris pour les vignettes"],
                 ["wp5_optimality_gap · wp5_joint_bound", "glouton vs optimum exact"],
                 ["wp10_sensitivity · wp10b_interaction", "les hypothèses balayées (80 réglages)"],
                 ["wp11_integration_demo", "chaîne complète sur vraies images JPEG"],
                 ["wp12_seams · wp12b_cut_recall · wp13_recompress_check",
                  "bords de tuiles et SAHI mesurés ; recompression côtière"],
                 ["build_report · build_overview · build_phase2_guide", "les PDF du dossier paper/"]],
                [6.2 * cm, W - 6.2 * cm]),
          P("<b>Relation entre les deux pipelines.</b> Le pipeline WP a établi les chiffres réels "
            "(détecteur, coûts en octets, ordonnanceur). Le pipeline du papier reprend son "
            "détecteur (YoloDetector charge les poids de wp1), son orbite (orbit.py) et ses mesures "
            "de SAHI (wp12) comme point de comparaison. Ses résultats deviendront RÉELS quand "
            "paper_b0_b4.py sera lancé avec --source airbus --detector yolo.", CALLOUT),
          PageBreak()]

    # ------------------------------------------------------------ 8. dashboard advice
    s += [P("8. Faut-il ajouter un tableau de bord ?", H1),
          P("<b>Recommandation : oui, mais au sol et en lecture seule.</b> Un tableau de bord qui lit "
            "les fichiers de code/results/ vaut l'effort. Il ne faut <b>pas</b> l'ajouter au pipeline "
            "embarqué : rien ne s'affiche à bord d'un satellite, et le papier ne le demande pas.",
            GOODBOX),
          P("Pourquoi oui", H2),
          *bullets([
              "<b>Pour le jury et la présentation</b> : le papier repose sur des compromis (octets "
              "contre rappel, énergie contre latence, direct contre relais). Un curseur de seuil qui "
              "fait bouger les courbes se comprend en dix secondes ; huit tableaux de chiffres, non.",
              "<b>Pour l'équipe</b> : voir d'un coup d'œil si une nouvelle exécution est SIM ou RÉELLE, "
              "si les paquets respectent l'échéance, et quelle ablation coûte le plus.",
              "<b>Coût faible</b> : toutes les données existent déjà (paper_b0_b4_sim.csv, "
              "paper_ablations_sim.csv, paper_tradeoff_sim.csv, paper_packets_sim.csv). Le tableau de "
              "bord ne recalcule rien. Compter un à deux jours."]),
          P("Les pièges à éviter", H2),
          *bullets([
              "<b>Ne jamais mélanger SIM et RÉEL</b> : bandeau permanent indiquant la source, comme "
              "sur les maquettes ci-dessous.",
              "<b>Ne pas recalculer dans le tableau de bord</b> : il lit les CSV ; pour de nouveaux "
              "chiffres, relancer paper_b0_b4.py. Sinon deux versions des résultats coexistent.",
              "<b>Ne pas cacher les hypothèses</b> : une page « Hypothèses » liste les puissances, "
              "débits et l'orbite du relais lus dans paper_b0_b4_sim.json."]),
          PageBreak()]

    s += [P("Maquettes", H1),
          P("Les trois pages ci-dessous sont des <b>maquettes</b> dessinées avec les vraies sorties de "
            "la campagne SIM (mêmes chiffres que la section 5) et la vraie géométrie des liaisons. "
            "Elles montrent ce que le tableau de bord afficherait ; elles ne sont pas des captures "
            "d'une application déjà construite."),
          fig("dashboard_1_vue_ensemble.png", W, "Page 1 — Vue d'ensemble : indicateurs clés de B4, "
              "octets transmis, qualité, énergie, effet de chaque ablation."),
          fig("dashboard_2_communication.png", W, "Page 2 — Communication : fenêtres de passage, "
              "fenêtres ISL, trajet de chaque paquet et latence selon le chemin."),
          fig("dashboard_3_compromis.png", W, "Page 3 — Compromis & scène : la scène avec la grille "
              "SAHI et les priorités P1/P2/P3, et les courbes éq. 32 et 34 au seuil choisi."),
          PageBreak()]

    # ------------------------------------------------------------ 9. how to build it
    s += [P("9. Comment le construire", H1),
          P("Outil conseillé", H2),
          P("<b>Streamlit + Plotly</b>, en Python. Le projet est déjà en Python et pandas lit les CSV ; "
            "Streamlit fournit les menus, curseurs et onglets sans écrire de HTML, et Plotly rend les "
            "graphiques interactifs (survol, zoom). Alternative sans serveur : une page HTML statique "
            "générée par un script, plus simple à partager mais sans curseur réel."),
          P("Architecture", H2),
          table([["Élément", "Contenu"],
                 ["code/dashboard/app.py", "point d'entrée : barre latérale (source, configuration, "
                  "seuil) et navigation"],
                 ["code/dashboard/data.py", "chargement des CSV et du JSON, avec cache ; choix SIM "
                  "(_sim) ou RÉEL (sans suffixe) selon les fichiers présents"],
                 ["code/dashboard/pages/", "1_vue_ensemble.py, 2_communication.py, 3_compromis.py, "
                  "4_ablations.py, 5_hypotheses.py"],
                 ["Sources", "paper_b0_b4*.csv, paper_ablations*.csv, paper_tradeoff*.csv, "
                  "paper_packets*.csv, paper_b0_b4*.json"]],
                [4.4 * cm, W - 4.4 * cm]),
          P("Étapes", H2),
          *bullets([
              "1. Installer : pip install streamlit plotly (dans .venv).",
              "2. Écrire data.py : une fonction par fichier, décorée de @st.cache_data.",
              "3. Page Vue d'ensemble : st.metric pour les 5 indicateurs, deux graphiques en barres.",
              "4. Page Communication : chronologie (barres horizontales des fenêtres et segments "
              "capture → livraison depuis paper_packets), nuage de latence, tableau des paquets.",
              "5. Page Compromis : st.select_slider sur les seuils du balayage ; marquer le point "
              "choisi sur les courbes éq. 32 et 34 ; afficher l'optimum de l'éq. 20 lu dans le JSON.",
              "6. Bandeau SIM/RÉEL et page Hypothèses lus dans le JSON.",
              "7. Lancer : streamlit run dashboard/app.py, puis ouvrir http://localhost:8501."]),
          P("Squelette de départ (non exécuté sur cette machine : Streamlit n'y est pas installé)", H2),
          Preformatted(
              "# code/dashboard/app.py\n"
              "import json\n"
              "from pathlib import Path\n"
              "import pandas as pd\n"
              "import plotly.express as px\n"
              "import streamlit as st\n\n"
              "RES = Path(__file__).resolve().parents[1] / \"results\"\n\n"
              "@st.cache_data\n"
              "def load(tag: str):\n"
              "    read = lambda n: pd.read_csv(RES / f\"{n}{tag}.csv\")\n"
              "    meta = json.loads((RES / f\"paper_b0_b4{tag}.json\").read_text())\n"
              "    return (read(\"paper_b0_b4\"), read(\"paper_ablations\"),\n"
              "            read(\"paper_tradeoff\"), read(\"paper_packets\"), meta)\n\n"
              "st.set_page_config(\"SAT7 Downlink\", layout=\"wide\")\n"
              "real = (RES / \"paper_b0_b4.csv\").exists()\n"
              "source = st.sidebar.radio(\"Source\", [\"RÉEL\", \"SIM\"] if real else [\"SIM\"])\n"
              "tab, abl, sw, pk, meta = load(\"\" if source == \"RÉEL\" else \"_sim\")\n"
              "if meta[\"label\"] == \"SIM\":\n"
              "    st.warning(\"Résultats SIM : scènes synthétiques, détecteur de secours.\")\n\n"
              "b4 = tab.set_index(\"config\").loc[\"B4\"]\n"
              "c = st.columns(5)\n"
              "c[0].metric(\"Réduction\", f\"x{b4.reduction_x:,.0f}\")\n"
              "c[1].metric(\"Rappel\", f\"{b4.recall:.2f}\")\n"
              "c[2].metric(\"Énergie économisée\", f\"{b4.ES:.1%}\")\n"
              "c[3].metric(\"Latence médiane\", f\"{b4.T_total_med_h:.2f} h\")\n"
              "c[4].metric(\"Part relais\", f\"{b4.relay_share:.0%}\")\n\n"
              "left, right = st.columns(2)\n"
              "left.plotly_chart(px.bar(tab, x=\"config\", y=\"D_tx_MB\", log_y=True))\n"
              "thr = st.sidebar.select_slider(\"Seuil\", options=list(sw.det_thr), value=0.25)\n"
              "fig = px.line(sw, x=\"D_tx_MB\", y=\"recall\", markers=True)\n"
              "sel = sw[sw.det_thr == thr]\n"
              "fig.add_scatter(x=sel.D_tx_MB, y=sel.recall, mode=\"markers\",\n"
              "                marker=dict(size=16, color=\"orange\"), name=f\"seuil {thr}\")\n"
              "right.plotly_chart(fig)\n"
              "st.dataframe(pk[[\"scene\", \"t_capture_h\", \"path\", \"bytes\", \"T_total\"]])", CODE),
          P("Ce squelette couvre la page Vue d'ensemble et le curseur de seuil ; les autres pages "
            "suivent le même modèle. Il faudra le tester une fois Streamlit installé.", SMALL)]

    out = PROJECT / "paper" / "IASTAM_P7_Guide_Pipeline_FR.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.7 * cm,
                            title="Guide du pipeline — IASTAM P7", author="ENIS")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", out)


if __name__ == "__main__":
    build()
