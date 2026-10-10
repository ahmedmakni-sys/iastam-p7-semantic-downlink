"""Energy, latency, data-reduction and joint-objective model (paper V, VI, VIII-E).

    E_total = E_proc + E_comm                                        (8), (30)
    E_proc  = E_pre + E_tile + E_det + E_fusion + E_semantic         (9)
    E_k     = P_k T_k                                                (10)
    E_proc  = sum_k P_k T_k                                          (11)
    E_comm,direct = P_tx T_tx,  T_tx = D_tx / R_tx                   (12)-(14)
    E_comm,relay  = E_ISL + E_GS                                     (15)
    E_comm  = min(E_comm,direct, E_comm,relay)  s.t. windows/deadline (16)  -> sat7.relay
    ES      = 1 - E_proposed / E_baseline                            (17)-(19)
    min a E_total + b D_tx + g T_total  s.t. Acc >= A_min, IP >= I_min   (20)-(22)
    DR      = 1 - D_tx / D_raw                                       (27)
    T_total = T_inference + T_encoding + T_communication + T_decoding    (28)
    T_communication = T_ISL + T_GS   (relay)                         (29)

The stage times T_k are MEASURED by the scripts that call this module (on the machine they run
on -- a laptop, not flight hardware). Every power P_k is an ASSUMPTION, kept in ``PowerConfig``
and printed with every result, so the energy figures are estimates and labelled as such.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

STAGES = ("pre", "tile", "det", "fusion", "semantic")


@dataclass
class PowerConfig:
    """Average power per stage, W. ASSUMPTIONS: a CubeSat OBC + an edge AI accelerator."""
    p_pre: float = 5.0           # on-board computer
    p_tile: float = 5.0
    p_det: float = 15.0          # Jetson-class accelerator under load
    p_fusion: float = 5.0
    p_semantic: float = 5.0
    p_tx: float = 12.0           # S-band downlink transmitter (RF + amplifier)
    p_isl: float = 25.0          # optical inter-satellite terminal (primary, transmitting)
    p_relay_gs: float = 12.0     # relay satellite's own downlink transmitter

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectorTiming:
    """A REAL per-inference detector time, to price E_det when this machine's own T_det is not
    the detector of interest (e.g. a SIM run with the classic fallback, or a CPU-only laptop)."""
    ms_per_inference: float
    source: str


def detector_timing(results_dir, which: str = "yolo_gpu") -> DetectorTiming:
    """Read a measured YOLOv8n inference time from the WP1 result files.

    ``yolo_gpu``: PyTorch on the RTX 5060 (wp1_metrics.json); ``yolo_cpu``: ONNX FP32 on the
    laptop CPU (wp1_export.json), the closer of the two to a CPU-only on-board computer.
    Idea from oblhdj/iastam-p7-semantic-downlink (EnergyModel.from_results, commit d69f85d).
    """
    import json
    from pathlib import Path
    d = Path(results_dir)
    if which == "yolo_gpu":
        m = json.loads((d / "wp1_metrics.json").read_text())
        return DetectorTiming(float(m["inference_ms_per_tile_gpu"]),
                              "REAL wp1_metrics.json inference_ms_per_tile_gpu (laptop RTX 5060)")
    if which == "yolo_cpu":
        x = json.loads((d / "wp1_export.json").read_text())
        return DetectorTiming(float(x["latency_ms_per_tile"]["onnx_fp32_cpu"]),
                              "REAL wp1_export.json onnx_fp32_cpu (laptop CPU)")
    raise ValueError(f"unknown detector timing {which!r}")


def stage_energy(power_w: float, time_s: float) -> float:
    """Eq. (10): E_k = P_k T_k."""
    return power_w * time_s


def processing_energy(times_s: dict, power: PowerConfig) -> dict:
    """Eqs. (9), (11): per-stage energies and their sum. ``times_s`` keys are ``STAGES``."""
    e = {k: stage_energy(getattr(power, f"p_{k}"), float(times_s.get(k, 0.0))) for k in STAGES}
    e["total"] = sum(e[k] for k in STAGES)
    return e


def transmission_time(d_bytes: float, rate_bps: float) -> float:
    """Eq. (13): T_tx = D_tx / R_tx (bytes over a bit rate)."""
    return 8.0 * d_bytes / rate_bps


def comm_energy_direct(d_bytes: float, rate_bps: float, p_tx: float) -> float:
    """Eq. (14): E_comm,direct = P_tx D_tx / R_tx."""
    return p_tx * transmission_time(d_bytes, rate_bps)


def comm_energy_relay(d_bytes: float, rate_isl_bps: float, rate_gs_bps: float,
                      power: PowerConfig) -> tuple[float, float, float]:
    """Eq. (15): E_comm,relay = E_ISL + E_GS. Returns (total, E_ISL, E_GS)."""
    e_isl = comm_energy_direct(d_bytes, rate_isl_bps, power.p_isl)
    e_gs = comm_energy_direct(d_bytes, rate_gs_bps, power.p_relay_gs)
    return e_isl + e_gs, e_isl, e_gs


def energy_saving(e_proposed: float, e_baseline: float) -> float:
    """Eq. (17): ES = 1 - E_proposed / E_baseline. Positive: onboard work pays for itself."""
    return 1.0 - e_proposed / e_baseline if e_baseline > 0 else float("nan")


def downlink_reduction(d_tx: float, d_raw: float) -> float:
    """Eq. (27): DR = 1 - D_tx / D_raw."""
    return 1.0 - d_tx / d_raw if d_raw > 0 else float("nan")


def total_latency(t_inference: float, t_encoding: float, t_communication: float,
                  t_decoding: float) -> float:
    """Eq. (28). For the relay path pass t_communication = T_ISL + T_GS (eq. 29)."""
    return t_inference + t_encoding + t_communication + t_decoding


@dataclass
class ObjectiveConfig:
    """Eq. (20)-(22). Terms are normalised by the raw baseline (B0) so the weights are unitless."""
    alpha: float = 1.0           # energy
    beta: float = 1.0            # transmitted data
    gamma: float = 1.0           # latency
    a_min: float = 0.5           # Accuracy >= A_min   (accuracy measured as F1)
    i_min: float = 0.9           # InformationPreservation >= I_min


def objective(e_total: float, d_tx: float, t_total: float, ref: dict,
              cfg: ObjectiveConfig) -> float:
    """Eq. (20): alpha E + beta D + gamma T, each term divided by its B0 reference value."""
    return (cfg.alpha * e_total / ref["E_total"] + cfg.beta * d_tx / ref["D_tx"]
            + cfg.gamma * t_total / ref["T_total"])


def feasible(accuracy: float, info_pres: float, cfg: ObjectiveConfig) -> bool:
    """Eqs. (21), (22)."""
    return accuracy >= cfg.a_min and info_pres >= cfg.i_min


def select_operating_point(rows: list[dict], ref: dict, cfg: ObjectiveConfig) -> dict | None:
    """Best feasible row under eq. (20). Rows need E_total, D_tx, T_total, F1, IP."""
    best, best_j = None, float("inf")
    for r in rows:
        if not feasible(r["F1"], r["IP"], cfg):
            continue
        j = objective(r["E_total"], r["D_tx"], r["T_total"], ref, cfg)
        if j < best_j:
            best, best_j = {**r, "J": j}, j
    return best
