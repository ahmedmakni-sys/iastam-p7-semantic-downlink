"""The paper's experimental campaign, B0 -> B1 -> B2 -> B3 -> B4 (eq. 31), plus its ablations.

"Energy-Aware Semantic Downlink for Maritime Object Detection with Optional Inter-Satellite
Relay" (IASTAM, ENIS) specifies the system below. This script runs it end to end and reports
every metric of section VIII-E:

    large image -> tiles 512 px, 20% overlap -> detector per tile (SAHI) -> global coordinates
    -> fusion -> semantic records (eq. 7) -> priorities P0..P3 (Table I) -> direct downlink
    or ISL relay (eqs. 16, 25, 26) -> ground decoding

  B0  no onboard detection             full raw image           raw-data reference
  B1  detector on the original image   detection output          detector-only reduction
  B2  SAHI + detector                  detection output          sliced inference
  B3  SAHI + detector + policy         metadata / ROI, direct    semantic downlink
  B4  SAHI + detector + policy         direct / optional relay   complete strategy

Ablations (paper VIII-G), all relative to B4: without SAHI, without tile overlap, without
fusion, without adaptive semantic downlink, without ROI transmission, without relay.
Trade-off curves (VIII-I): D_tx vs recall (32), E_total vs T_total (33), D_tx vs information
preservation (34), (E_direct, T_direct) vs (E_relay, T_relay) (35).

The joint objective (20)-(22) is solved per object (``sat7.joint``): "B4+J" is B4 with every
detection's level P0-P3 chosen by the program instead of by confidence thresholds, and
paper_joint.csv sweeps its weights and its information floor.

What is MEASURED vs ASSUMED, printed on every run:
  measured  detections, bytes of every packet (real serialisation + real JPEG ROIs), every
            stage time T_k on THIS machine (or the REAL YOLOv8n time, --det-timing),
            orbit passes and ISL geometry (SGP4)
  assumed   every power P_k, link rates (orbit.LinkConfig, relay.RelayConfig), capture times
            (spread over the day), the deadline, the eq. (20)/(25)/(26) weights and floors

Sources (--source):
  synthetic  generated scenes, no data needed (SIM)
  swaths     the 24 real 3072 px swaths of results/oblhdj/wp15_manifest.json -- the scenes on
             which oblhdj/iastam-p7-semantic-downlink measured SAHI, so results compare 1:1
  airbus     random 2 x 2 mosaics of real test tiles
  dota, coco the paper's recommended large-scene sets (DOTA; HRSID as COCO)

    # anywhere: synthetic scenes + classic fallback detector, E_det on the REAL YOLO CPU time
    .venv/Scripts/python.exe scripts/paper_b0_b4.py --det-timing yolo_cpu
    # the real campaign (GPU machine with data/ and runs/): REAL B0-B4 on the real swaths
    .venv312/Scripts/python.exe scripts/paper_b0_b4.py --source swaths --detector yolo
    # the paper's 768 vs 512 question, two ground stations
    .venv312/Scripts/python.exe scripts/paper_b0_b4.py --source swaths --detector yolo \
        --tile 768 --stations sfax,svalbard --tag _swaths768

Outputs: results/paper_b0_b4.csv|.json, paper_ablations.csv, paper_tradeoff.csv,
         paper_joint.csv|.png, paper_packets.csv, paper_links.csv, paper_scene.jpg|.csv,
         paper_tradeoffs.png   (suffix _sim for a synthetic SIM run, --tag to choose)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sat7.datasets import iter_coco, iter_dota
from sat7.detectors import ClassicDetector
from sat7.energy import (ObjectiveConfig, PowerConfig, detector_timing, downlink_reduction,
                         energy_saving, processing_energy, select_operating_point, total_latency)
from sat7.joint import (P1, CostModel, JointConstraints, JointWeights, assemble_packet,
                        constraint_sweep,
                        encode_candidates, pareto, solve)
from sat7.metrics import DetAccumulator, preserved
from sat7.orbit import LINK_PRESETS, STATION_PRESETS, OrbitConfig
from sat7.pipeline import (OnboardPipeline, PipelineConfig, ground_decode, preprocess,
                           record_centres)
from sat7.relay import (Network, PathPolicy, RelayConfig, ground_windows, isl_windows,
                        relay_orbit)
from sat7.sahi import SahiConfig
from sat7.scenes import airbus_mosaic, swaths_from_manifest, synthetic_scene
from sat7.semantic import PriorityPolicy

SWEEP = (0.10, 0.15, 0.25, 0.35, 0.45, 0.55, 0.70)
JOINT = "B4+J"                     # B4 with eq. (20) choosing each object's level


def experiment_plan(args, det_ms=None):
    """Every configuration of the paper as (PipelineConfig, or None for raw; relay allowed)."""
    s = SahiConfig(args.tile, args.overlap, args.fusion, conf_thr=args.conf)
    pol = PriorityPolicy(t_discard=args.conf)

    def cfg(**kw):
        return PipelineConfig(**{"slicing": s, "policy": pol, "det_ms": det_ms, **kw})

    campaign = {                                                  # Table II, eq. (31)
        "B0": (None, False),
        "B1": (cfg(sahi=False, semantic=False), False),
        "B2": (cfg(semantic=False), False),
        "B3": (cfg(), False),
        "B4": (cfg(), True),
    }
    ablations = {                                                 # VIII-G
        "B4 full": (cfg(), True),
        "without SAHI": (cfg(sahi=False), True),
        "without tile overlap": (cfg(slicing=replace(s, overlap=0.0)), True),
        "without fusion": (cfg(slicing=replace(s, fusion="none")), True),
        "without adaptive downlink": (cfg(policy=replace(pol, adaptive=False)), True),
        "without ROI transmission": (cfg(policy=replace(pol, roi=False)), True),
        "without relay": (cfg(), False),
    }
    return campaign, ablations


def make_detector(args):
    if args.detector == "yolo":
        from sat7.detectors import YoloDetector
        return YoloDetector(args.weights, imgsz=args.imgsz, conf=0.05)
    return ClassicDetector(input_size=args.imgsz)


def iter_scenes(args, rng):
    """Yield (index, name, image, ground truth) for the chosen source, at most --scenes of them."""
    if args.source == "synthetic":
        for i in range(args.scenes):
            # ASSUMED sky: 70% of scenes clear, the rest 5-20% cloud (clouds cause false alarms)
            cloud = 0.0 if rng.random() < 0.7 else float(rng.uniform(0.05, 0.20))
            img, gts = synthetic_scene(args.scene_px, int(rng.integers(10, 30)), args.seed + i, cloud)
            yield i, f"synthetic_{i:03d}", img, gts
        return
    if args.source in ("swaths", "dota", "coco"):
        if args.source == "swaths":
            if not (args.data / "images" / "test").is_dir():
                sys.exit(f"--source swaths needs the Airbus test split under {args.data} "
                         "(images/test, labels/test): build it with scripts/wp0_build_dataset.py, "
                         "or run on the GPU machine that has data/yolo_ships")
            it = swaths_from_manifest(args.manifest, args.data)
        elif args.source == "dota":
            it = iter_dota(args.dota_images, args.dota_labels)
        else:
            it = iter_coco(args.coco_json, args.coco_images)
        for i, (name, img, gts) in enumerate(it):
            if i >= args.scenes:
                return
            yield i, name, img, gts
        return
    img_dir = args.data / "images" / "test"
    lbl_dir = args.data / "labels" / "test"
    files = sorted(img_dir.glob("*.jpg"))
    if not files:
        sys.exit(f"no test images under {img_dir} -- build the dataset with wp0 first")
    k2 = args.mosaic_k ** 2
    pick = rng.choice(len(files), size=min(args.scenes * k2, len(files)), replace=False)
    for i in range(len(pick) // k2):
        group = [files[j] for j in pick[i * k2:(i + 1) * k2]]
        img, gts = airbus_mosaic(group, lbl_dir, args.mosaic_k)
        yield i, f"mosaic_{i:03d}", img, gts


def save_example_scene(out_dir: Path, tag: str, img, gts, pipe, detected, max_px: int = 1024) -> dict:
    """One scene, downscaled, with its B4 detections and their priorities (dashboard scene view).

    Boxes stay in full-resolution scene pixels; the returned dict carries the scale.
    """
    out_dir.mkdir(exist_ok=True)
    s = min(1.0, max_px / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / f"paper_scene{tag}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    dets, _ = pipe.detections(detected)
    rows = [{"kind": "detection", "x": d.x, "y": d.y, "w": d.w, "h": d.h, "p": d.p,
             "priority": pipe.cfg.policy.priority(d.p)} for d in dets]
    rows += [{"kind": "ground_truth", "x": g.x, "y": g.y, "w": g.w, "h": g.h, "p": 1.0,
              "priority": -1} for g in gts]
    pd.DataFrame(rows).to_csv(out_dir / f"paper_scene{tag}.csv", index=False)
    return {"width": int(img.shape[1]), "height": int(img.shape[0]), "scale": s,
            "tile": pipe.cfg.slicing.tile, "overlap": pipe.cfg.slicing.overlap}


class Evaluation:
    """One configuration over all scenes: its packets and its detection metrics."""

    def __init__(self):
        self.at_thr = DetAccumulator()      # precision / recall / F1 at the threshold
        self.full = DetAccumulator()        # every confidence, for mAP
        self.per_scene: dict[int, dict] = {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "swaths", "airbus", "dota", "coco"],
                    default="synthetic",
                    help="swaths = the 24 real 3072 px swaths of results/oblhdj/wp15_manifest.json")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo_ships")
    ap.add_argument("--manifest", type=Path, default=ROOT / "results" / "oblhdj" / "wp15_manifest.json")
    ap.add_argument("--dota-images", type=Path, help="DOTA images/ directory")
    ap.add_argument("--dota-labels", type=Path, help="DOTA labelTxt/ directory")
    ap.add_argument("--coco-json", type=Path, help="COCO annotations, e.g. HRSID test2017.json")
    ap.add_argument("--coco-images", type=Path, help="directory of the COCO images")
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--scene-px", type=int, default=2048, help="synthetic scene side")
    ap.add_argument("--mosaic-k", type=int, default=2, help="airbus: k x k tiles per scene")
    ap.add_argument("--detector", choices=["classic", "yolo"], default="classic")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "ships" / "weights" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--fusion", choices=["nmm", "nms"], default="nmm")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--capture-hours", type=float, default=24.0)
    ap.add_argument("--horizon-hours", type=float, default=48.0)
    ap.add_argument("--deadline-hours", type=float, default=6.0)
    ap.add_argument("--link", default="cubesat_sband", choices=list(LINK_PRESETS))
    ap.add_argument("--stations", default="sfax",
                    help=f"comma-separated ground stations among {', '.join(STATION_PRESETS)}")
    ap.add_argument("--det-timing", choices=["measured", "yolo_gpu", "yolo_cpu"], default="measured",
                    help="price E_det on this run's own time, or on the REAL YOLOv8n time (wp1)")
    ap.add_argument("--joint-floor", type=float, default=0.10,
                    help="eq. (20) decides P0-P3 for every detection above this confidence")
    ap.add_argument("--a-min", type=float, default=0.90, help="eq. (21) accuracy floor")
    ap.add_argument("--i-min", type=float, default=0.50, help="eq. (22) information floor")
    ap.add_argument("--weights-abg", default="1,1,1", help="alpha,beta,gamma of eq. (20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="output suffix (default: '' if REAL, else _sim...)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    label = "REAL" if (args.source != "synthetic" and args.detector == "yolo") else "SIM"
    tag = args.tag if args.tag is not None else (
        "" if label == "REAL" else "_sim" if args.source == "synthetic" else f"_sim_{args.source}")
    rng = np.random.default_rng(args.seed)
    det = make_detector(args)
    power = PowerConfig()
    timing = None if args.det_timing == "measured" else detector_timing(ROOT / "results",
                                                                        args.det_timing)
    campaign, ablations = experiment_plan(args, timing.ms_per_inference if timing else None)
    pipes = {name: OnboardPipeline(det, c) for name, (c, _) in {**campaign, **ablations}.items()
             if c is not None}
    for t in SWEEP:                                   # eqs. (32), (34): B4 at other thresholds
        pipes[f"thr {t:.2f}"] = pipes["B4"].with_threshold(t)
    evals = {name: Evaluation() for name in [*pipes, JOINT]}
    epoch = datetime(2026, 9, 18, tzinfo=timezone.utc)
    print(f"=== Paper B0-B4 campaign [{label}]: source={args.source}, detector={det.name}, "
          f"tiles {args.tile}px @ {args.overlap:.0%} overlap, fusion={args.fusion}, "
          f"conf>={args.conf}, E_det timing={args.det_timing} ===")

    # ---------------- links first: eq. (20) prices bytes at the real mean downlink rate
    stations = [STATION_PRESETS[s.strip()] for s in args.stations.split(",") if s.strip()]
    link = LINK_PRESETS[args.link]
    prim = OrbitConfig(epoch=epoch)
    rcfg = RelayConfig()
    rel = relay_orbit(prim, rcfg)
    hz = args.horizon_hours
    w_direct = ground_windows(prim, stations, link, epoch, hz)
    w_isl = isl_windows(prim, rel, epoch, hz, rcfg)
    w_relay = ground_windows(rel, stations, link, epoch, hz)
    r_mean = (sum((w.end - w.start) * w.rate_bps for w in w_direct)
              / max(sum(w.end - w.start for w in w_direct), 1e-9))
    cost = CostModel(p_tx_W=power.p_tx, r_tx_bps=r_mean, p_proc_W=power.p_semantic)
    alpha, beta, gamma = (float(v) for v in args.weights_abg.split(","))
    jweights = JointWeights(alpha, beta, gamma)
    jcons = JointConstraints(args.a_min, args.i_min)
    print(f"links over {hz:.0f} h, stations {[s.name for s in stations]}: {len(w_direct)} primary "
          f"windows, {len(w_isl)} ISL windows ({sum(w.end - w.start for w in w_isl) / 60:.0f} min), "
          f"{len(w_relay)} relay windows; mean downlink {r_mean / 1e6:.2f} Mbit/s\n")

    # ---------------- onboard: every configuration on every scene
    scenes = []
    joint_objs, joint_enc, joint_scene = [], {}, {}
    for iid, scene_name, image, gts in iter_scenes(args, rng):
        img, t_pre = preprocess(image)
        t_cap = float(rng.uniform(0, args.capture_hours * 3600))
        ts = epoch.timestamp() + t_cap
        detected = {}                                 # the detector runs once per slicing
        for name, pipe in pipes.items():
            key = pipe.cfg.detection_key
            if key not in detected:
                detected[key] = pipe.detect(img)
            out = pipe.run(img, iid, ts, detected[key])
            pkt, t_dec = ground_decode(out.packet)
            ev = evals[name]
            ev.at_thr.add(out.detections, gts)
            ev.full.add(pipe.detections(detected[key], conf_thr=0.0)[0], gts)
            ev.per_scene[iid] = {"bytes": len(out.packet), "times": out.times, "t_dec": t_dec,
                                 "cents": record_centres(pkt)}
        b2, b4 = pipes["B2"], pipes["B4"]
        reference = b2.detections(detected[b2.cfg.detection_key])[0]   # ground analysis of B0
        # eq. (20) candidates: every detection above the floor, each level measured on the image
        cands = b4.detections(detected[b4.cfg.detection_key], conf_thr=args.joint_floor)[0]
        objs, enc = encode_candidates(img, cands, cost, b4.cfg.roi, first_id=len(joint_enc))
        joint_objs += objs
        joint_enc.update(enc)
        joint_scene[iid] = {"ids": [o.id for o in objs], "gts": gts, "ts": ts,
                            "base_times": {k: v for k, v in evals["B4"].per_scene[iid]["times"].items()
                                           if k != "semantic"}}
        if not scenes:                                 # one example scene for the dashboard
            example = save_example_scene(args.out, tag, img, gts, b4,
                                         detected[b4.cfg.detection_key])
            example["name"] = scene_name
        scenes.append({"iid": iid, "name": scene_name, "t_cap": t_cap, "t_pre": t_pre,
                       "raw_bytes": img.nbytes, "reference": reference})
        print(f"  {scene_name}: {img.shape[1]}x{img.shape[0]}, {len(gts):3d} ships, "
              f"direct {len(evals['B1'].per_scene[iid]['cents']):3d} / "
              f"SAHI {len(reference):3d} detections, {len(objs)} eq.20 candidates", flush=True)

    def joint_eval(name: str, weights: JointWeights, constraints: JointConstraints | None = None):
        """Solve eq. (20) over every candidate of the run, then build each scene's packet."""
        res = solve(joint_objs, weights, constraints or jcons)
        ev = evals.setdefault(name, Evaluation())
        ev.full = evals["B4"].full
        for iid, js in joint_scene.items():
            t0 = time.perf_counter()
            pkt = assemble_packet(js["ids"], res.assignment, joint_enc, iid, js["ts"])
            data = pkt.encode()
            t_sem = time.perf_counter() - t0 + sum(
                joint_enc[o].encode_time_s(res.assignment[o]) for o in js["ids"])
            back, t_dec = ground_decode(data)
            reported = [joint_enc[o].detection for o in js["ids"] if res.assignment[o] >= P1]
            ev.at_thr.add(reported, js["gts"])
            ev.per_scene[iid] = {"bytes": len(data), "times": {**js["base_times"], "semantic": t_sem},
                                 "t_dec": t_dec, "cents": record_centres(back)}
        return res

    jres = joint_eval(JOINT, jweights)
    gap_txt = (f"gap {jres.gap:.1%}" if np.isfinite(jres.gap)
               else "the free optimum sends nothing, so the floors set the cost")
    print(f"\neq. (20) on {len(joint_objs)} objects: J {jres.J:.4g} (lower bound "
          f"{jres.J_lower_bound:.4g}, {gap_txt}), accuracy {jres.accuracy:.3f} >= "
          f"{jcons.a_min}, information {jres.info_preservation:.3f} >= {jcons.i_min}, "
          f"feasible={jres.feasible}, valid={jres.valid}")

    def transmit(name: str | None, relay: bool) -> pd.DataFrame:
        """Send one configuration's packets, in capture order, over the network."""
        net = Network(w_direct, w_isl, w_relay, power,
                      PathPolicy(allow_relay=relay, deadline_s=args.deadline_hours * 3600))
        rows = []
        for s in sorted(scenes, key=lambda z: z["t_cap"]):
            if name is None:                                   # B0: the raw image itself
                nbytes, times, t_dec, cents = s["raw_bytes"], {"pre": s["t_pre"]}, 0.0, None
            else:
                r = evals[name].per_scene[s["iid"]]
                nbytes, times, t_dec, cents = r["bytes"], r["times"], r["t_dec"], r["cents"]
            t_enc = times.get("semantic", 0.0)
            t_inf = sum(v for k, v in times.items() if k != "semantic")
            ready = s["t_cap"] + t_inf + t_enc
            ch = net.send(ready, nbytes)
            ok = ch.path != "none" and ch.finish <= hz * 3600
            n_ref = len(s["reference"])
            if not ok:
                pres = 0
            elif cents is None:
                pres = n_ref                                   # B0: the ground has the image
            else:
                pres = sum(preserved(s["reference"], cents))
            rows.append({"scene": s["iid"], "t_capture_h": s["t_cap"] / 3600,
                         "t_ready_h": ready / 3600,
                         "t_delivered_h": ch.finish / 3600 if ok else np.nan,
                         "path": ch.path, "ok": ok, "bytes": nbytes, "raw": s["raw_bytes"],
                         "E_proc": processing_energy(times, power)["total"],
                         "E_comm": ch.e_comm if ok else 0.0,
                         "T_total": total_latency(t_inf, t_enc, ch.t_comm, t_dec) if ok else np.nan,
                         "pres": pres, "n_ref": n_ref, "opts": ch.options})
        return pd.DataFrame(rows)

    def summarise(name: str, df: pd.DataFrame, det_name: str, thr: float) -> dict:
        ev = evals[det_name]
        m = ev.at_thr.at(thr)
        d_tx, d_raw = df.bytes.sum(), df.raw.sum()
        e_tot = df.E_proc.sum() + df.E_comm.sum()
        return {"config": name, "D_tx_MB": d_tx / 1e6, "DR": downlink_reduction(d_tx, d_raw),
                "reduction_x": d_raw / d_tx, "precision": m["precision"], "recall": m["recall"],
                "F1": m["F1"], "mAP50": ev.full.ap(), "small_recall": ev.at_thr.small_recall(),
                "IP": df.pres.sum() / max(df.n_ref.sum(), 1),
                "E_proc_J": df.E_proc.sum(), "E_comm_J": df.E_comm.sum(), "E_total_J": e_tot,
                "T_total_med_h": float(np.nanmedian(df.T_total)) / 3600,
                "T_total_mean_h": float(np.nanmean(df.T_total)) / 3600,
                "delivered": df.ok.mean(), "relay_share": (df.path == "relay").mean(),
                "D_tx": d_tx, "E_total": e_tot, "T_total": float(np.nanmean(df.T_total))}

    frames, rows = {}, []
    for name, (c, relay) in campaign.items():
        frames[name] = transmit(None if c is None else name, relay)
        # B0's detection quality: the ground runs the same SAHI detector on the raw image
        rows.append(summarise(name, frames[name], "B2" if c is None else name, args.conf))
    frames[JOINT] = transmit(JOINT, True)                     # B4, levels chosen by eq. (20)
    rows.append(summarise(JOINT, frames[JOINT], JOINT, 0.0))
    tab = pd.DataFrame(rows)
    b0 = tab.iloc[0]
    tab["ES"] = [energy_saving(e, b0.E_total_J) for e in tab.E_total_J]

    abl = pd.DataFrame([summarise(name, transmit(name, relay), name, args.conf)
                        for name, (_, relay) in ablations.items()])
    abl["ES"] = [energy_saving(e, b0.E_total_J) for e in abl.E_total_J]

    sw = pd.DataFrame([{"det_thr": t, **summarise(f"thr {t:.2f}", transmit(f"thr {t:.2f}", True),
                                                   f"thr {t:.2f}", t)} for t in SWEEP])

    ref = {"E_total": b0.E_total, "D_tx": b0.D_tx, "T_total": b0.T_total}
    obj = ObjectiveConfig()
    best = select_operating_point(sw.to_dict("records"), ref, obj)

    # eq. (20) at several (alpha, beta, gamma): the program's own Pareto corners, each sent
    # over the real network so D, E and T are the delivered ones, not the program's estimate
    par = []
    runs = ([("weights", w, jcons) for w, _ in pareto(joint_objs, jcons)]
            + [("i_min", jweights, c) for c, _ in constraint_sweep(joint_objs, jweights, jcons.a_min)])
    for k, (sweep, w, c) in enumerate(runs):
        r = joint_eval(f"J{k}", w, c)
        s_ = summarise(f"J{k}", transmit(f"J{k}", True), f"J{k}", 0.0)
        par.append({"sweep": sweep, "alpha": w.alpha, "beta": w.beta, "gamma": w.gamma,
                    "a_min": c.a_min, "i_min": c.i_min,
                    "J": r.J, "J_lower_bound": r.J_lower_bound, "gap": r.gap,
                    "accuracy_eq21": r.accuracy, "information_eq22": r.info_preservation,
                    "feasible": r.feasible, "valid": r.valid,
                    "P1": sum(v == 1 for v in r.assignment.values()),
                    "P2": sum(v == 2 for v in r.assignment.values()),
                    "P3": sum(v == 3 for v in r.assignment.values()),
                    **{col: s_[col] for col in ("D_tx_MB", "E_total_J", "T_total_med_h",
                                                "recall", "precision", "IP")}})
    par = pd.DataFrame(par)

    # ---------------- report
    pd.set_option("display.width", 220)
    show = ["config", "D_tx_MB", "reduction_x", "DR", "precision", "recall", "F1", "mAP50",
            "small_recall", "IP", "E_proc_J", "E_comm_J", "E_total_J", "ES", "T_total_med_h",
            "delivered", "relay_share"]

    def fmt(v):
        return f"{v:.4g}"

    print(f"\n=== B0 -> B4 [{label}] ===")
    print(tab[show].to_string(index=False, float_format=fmt))
    print("  (B0 detection quality = the ground running the same SAHI detector on the raw image)")
    print(f"\n=== Ablations (VIII-G) [{label}] ===")
    print(abl[show].to_string(index=False, float_format=fmt))
    print(f"\n=== Confidence sweep on B4: eqs. (32), (34) [{label}] ===")
    print(sw[["det_thr", "D_tx_MB", "precision", "recall", "F1", "IP", "E_total_J",
              "T_total_med_h"]].to_string(index=False, float_format=fmt))
    if best:
        print(f"\nThreshold selection under eq. (20), F1 >= {obj.a_min}, IP >= {obj.i_min}: "
              f"best threshold {best['det_thr']:.2f}  (J = {best['J']:.4f})")
    else:
        print(f"\nThreshold selection: no sweep point satisfies F1 >= {obj.a_min} and IP >= {obj.i_min}")
    print(f"\n=== eq. (20) solved per object (B4+J), Pareto corners; A_min {jcons.a_min}, "
          f"I_min {jcons.i_min} [{label}] ===")
    print(par.to_string(index=False, float_format=fmt))
    print("\nASSUMED (not measured): powers", power.as_dict())
    print(f"  link {args.link}, stations {args.stations}, ISL {rcfg.isl_rate_bps/1e6:.0f} Mbps up to "
          f"{rcfg.isl_max_range_km:.0f} km, relay RAAN +{rcfg.raan_offset_deg:.0f} deg, deadline "
          f"{args.deadline_hours} h, capture times uniform over {args.capture_hours} h")
    if timing:
        print(f"  E_det priced on {timing.ms_per_inference:.2f} ms/inference: {timing.source}")
    if label == "SIM":
        print("  SIM: synthetic scenes and/or the classic fallback detector -- not YOLO results")

    # ---------------- save
    args.out.mkdir(exist_ok=True)
    internal = ["D_tx", "E_total", "T_total"]
    tab.drop(columns=internal).to_csv(args.out / f"paper_b0_b4{tag}.csv", index=False)
    abl.drop(columns=internal).to_csv(args.out / f"paper_ablations{tag}.csv", index=False)
    sw.drop(columns=internal + ["config"]).to_csv(args.out / f"paper_tradeoff{tag}.csv", index=False)
    par.to_csv(args.out / f"paper_joint{tag}.csv", index=False)
    frames["B4"].drop(columns=["opts"]).to_csv(args.out / f"paper_packets{tag}.csv", index=False)
    pd.DataFrame([{"link": n, "start_h": w.start / 3600, "end_h": w.end / 3600,
                   "rate_Mbps": w.rate_bps / 1e6}
                  for n, ws in (("direct", w_direct), ("isl", w_isl), ("relay_gs", w_relay))
                  for w in ws]).to_csv(args.out / f"paper_links{tag}.csv", index=False)
    (args.out / f"paper_b0_b4{tag}.json").write_text(json.dumps({
        "label": label, "source": args.source, "detector": det.name, "scenes": len(scenes),
        "scene_names": [s["name"] for s in scenes],
        "sahi": vars(campaign["B4"][0].slicing), "powers_W_ASSUMED": power.as_dict(),
        "det_timing": ({"mode": args.det_timing, "ms_per_inference": timing.ms_per_inference,
                        "source": timing.source} if timing else {"mode": "measured"}),
        "relay_ASSUMED": vars(rcfg), "link": args.link, "deadline_h": args.deadline_hours,
        "stations": [s.name for s in stations], "mean_downlink_Mbps": r_mean / 1e6,
        "objective": vars(obj), "best_operating_point": best, "example_scene": example,
        "joint_eq20": {"weights": vars(jweights), "constraints": vars(jcons),
                       "candidate_floor": args.joint_floor, "objects": len(joint_objs),
                       "J": jres.J, "J_lower_bound": jres.J_lower_bound, "gap": jres.gap,
                       "accuracy": jres.accuracy, "information": jres.info_preservation,
                       "feasible": jres.feasible, "valid": jres.valid,
                       "levels": {f"P{k}": sum(v == k for v in jres.assignment.values())
                                  for k in range(4)}},
        "capture_hours": args.capture_hours, "horizon_hours": args.horizon_hours,
        "links": {"primary_passes": len(w_direct), "isl_windows": len(w_isl),
                  "relay_passes": len(w_relay)},
    }, indent=2, default=float))

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    a = ax[0, 0]
    a.plot(sw.D_tx_MB, sw.recall, "o-", color="#1B6CA8")
    for _, r in sw.iterrows():
        a.annotate(f"{r.det_thr:.2f}", (r.D_tx_MB, r.recall), fontsize=7, xytext=(4, -8),
                   textcoords="offset points")
    a.set(xlabel="D_tx (MB)", ylabel="recall", title="(32) D_tx vs recall, B4 threshold sweep")
    a = ax[0, 1]
    for _, r in tab.iterrows():
        a.scatter(r.E_total_J, r.T_total_med_h, s=60)
        a.annotate(r.config, (r.E_total_J, r.T_total_med_h), fontsize=8, xytext=(4, 4),
                   textcoords="offset points")
    a.set(xscale="log", xlabel="E_total (J, log)", ylabel="median T_total (h)",
          title="(33) E_total vs T_total, B0-B4")
    a = ax[1, 0]
    a.plot(sw.D_tx_MB, sw.IP, "o-", color="#2E8B57")
    a.set(xlabel="D_tx (MB)", ylabel="information preservation",
          title="(34) D_tx vs information preservation")
    a = ax[1, 1]
    for k, c in (("direct", "#C0392B"), ("relay", "#7D3C98")):
        pts = [o[k] for o in frames["B4"].opts if k in o]
        if pts:
            e, t = zip(*pts)
            a.scatter(e, np.array(t) / 3600, s=22, color=c, label=k, alpha=0.8)
    a.set(xlabel="E_comm per packet (J)", ylabel="T_comm (h)",
          title="(35) (E_direct, T_direct) vs (E_relay, T_relay)")
    a.legend(fontsize=8, frameon=False)
    for a in ax.flat:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Paper trade-off curves [{label}]", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / f"paper_tradeoffs{tag}.png", dpi=150)
    plt.close(fig)

    ims = par[par.sweep == "i_min"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    a0.plot(ims.D_tx_MB, ims.information_eq22, "o-", color="#2E8B57")
    for _, r in ims.iterrows():
        a0.annotate(f"I_min {r.i_min:.1f}", (r.D_tx_MB, r.information_eq22), fontsize=7,
                    xytext=(4, -10), textcoords="offset points")
    a0.set(xlabel="D_tx (MB)", ylabel="information preserved, eq. (22)",
           title="eq. (20) solved per object: bytes bought by each information floor")
    x = np.arange(len(ims))
    bottom = np.zeros(len(ims))
    for lv, c in (("P1", "#2E8B57"), ("P2", "#E07A1F"), ("P3", "#C0392B")):
        a1.bar(x, ims[lv], bottom=bottom, color=c, label=lv)
        bottom += ims[lv].to_numpy()
    a1.set_xticks(x, [f"{v:.1f}" for v in ims.i_min])
    a1.set(xlabel="I_min", ylabel="objects", title="levels chosen by the solver")
    a1.legend(fontsize=8, frameon=False)
    for a in (a0, a1):
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Joint objective, eqs. (20)-(22) [{label}]", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / f"paper_joint{tag}.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved paper_b0_b4{tag}.csv|.json, paper_ablations{tag}.csv, paper_tradeoff{tag}.csv, "
          f"paper_joint{tag}.csv, paper_packets{tag}.csv, paper_links{tag}.csv, "
          f"paper_tradeoffs{tag}.png in {args.out}")


if __name__ == "__main__":
    main()
