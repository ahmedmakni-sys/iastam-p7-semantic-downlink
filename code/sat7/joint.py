"""The paper's joint objective, solved as one program (section VI, eqs. 20-22).

    minimise   J = alpha * E_total + beta * D_tx + gamma * T_total
    subject to Accuracy                >= A_min                                   (21)
               InformationPreservation >= I_min                                   (22)

Decision: for every detection j, a priority level x_j in {P0, P1, P2, P3} (Table I). Raising a
level costs bytes, energy and latency and earns accuracy (the object is reported) and information
(how richly). With aggregate constraints this is a multiple-choice knapsack:

    Accuracy                = sum_j w_j [x_j >= P1] / sum_j w_j
    InformationPreservation = sum_j info_j(x_j)     / sum_j info_j(P3)

with w_j the detection's confidence (its expected chance of being a real ship) and info credits
P0 0, P1 0.5, P2 0.8, P3 1.0 (ASSUMPTION, swept like the other weights).

Solver: the per-object unconstrained argmin of J is a true lower bound (the constraints only ever
push levels up); a greedy repair then lifts objects, cheapest extra J per unit of deficit first,
until both floors hold or the instance is proven unreachable. ``JointResult.gap`` is the price
the constraints add over that bound.

The solver is ported from oblhdj/iastam-p7-semantic-downlink (``sat7/joint.py``, commit d69f85d,
MIT). Two changes:
  * units: a link rate is in bits/s, so a level's transmit time is 8 * bytes / R. The original
    divided bytes by bits/s, which made E_comm and T_comm 8 times too small;
  * costs are measured, not modelled: each level's bytes are this pipeline's real serialised
    record + real JPEG ROI/context, and its processing time is the measured encoding time.

Labels: D (bytes) REAL serialisation; T and E use the link rate and powers in ``CostModel``
(SIM / ASSUMPTION); alpha, beta, gamma, A_min, I_min are ASSUMPTION, swept by ``pareto``.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from .sahi import Detection
from .semantic import (_BLOB, P0, P1, P2, P3, RoiConfig, SemanticPacket, SemanticRecord,
                       extract_context, extract_roi)

INFO_CREDIT = {P0: 0.0, P1: 0.5, P2: 0.8, P3: 1.0}
MB = 1e6
HOUR = 3600.0


@dataclass
class JointWeights:
    """alpha, beta, gamma of eq. (20), per joule, per megabyte, per hour. ASSUMPTION, swept."""
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0


@dataclass
class JointConstraints:
    """A_min and I_min of eqs. (21)-(22). ASSUMPTION, swept."""
    a_min: float = 0.90
    i_min: float = 0.50


@dataclass
class CostModel:
    """Bytes -> energy and latency for one object's level (eqs. 10-14)."""
    p_tx_W: float = 12.0          # transmitter power (sat7.energy.PowerConfig.p_tx)
    r_tx_bps: float = 2.53e6      # effective downlink rate, bits/s (orbit sim mean)
    p_proc_W: float = 5.0         # on-board computer power while encoding (PowerConfig.p_semantic)

    def tx_time_s(self, nbytes: float) -> float:
        return 8.0 * nbytes / self.r_tx_bps

    def energy_J(self, nbytes: float, t_proc_s: float) -> float:
        return self.p_proc_W * t_proc_s + self.p_tx_W * self.tx_time_s(nbytes)

    def latency_s(self, nbytes: float, t_proc_s: float) -> float:
        """Per-item lower bound: encoding + transmission, no queueing (relay.py does that)."""
        return t_proc_s + self.tx_time_s(nbytes)


@dataclass
class LevelOption:
    level: int
    bytes: float
    energy_J: float
    latency_s: float
    acc: float                    # accuracy credit: the weight if reported (level >= P1)
    info: float                   # information credit: weight * INFO_CREDIT[level]


@dataclass
class JointObject:
    id: int
    weight: float
    options: list[LevelOption]
    info_max: float


@dataclass
class EncodedDetection:
    """Everything needed to build this detection's record at any level, without the image."""
    detection: Detection
    roi: bytes
    context: bytes
    t_roi_s: float = 0.0          # measured encoding times, so E_semantic follows the assignment
    t_context_s: float = 0.0

    def encode_time_s(self, level: int) -> float:
        return (self.t_roi_s if level >= P2 else 0.0) + (self.t_context_s if level >= P3 else 0.0)


def make_object(oid: int, weight: float, sizes: dict, proc_s: dict,
                cost: CostModel) -> JointObject:
    """One object from its byte size and processing time at each level."""
    opts = [LevelOption(lv, sizes[lv], cost.energy_J(sizes[lv], proc_s.get(lv, 0.0)),
                        cost.latency_s(sizes[lv], proc_s.get(lv, 0.0)),
                        weight if lv >= P1 else 0.0, weight * INFO_CREDIT[lv])
            for lv in sorted(sizes)]
    return JointObject(oid, weight, opts, max(o.info for o in opts))


def encode_candidates(img: np.ndarray, dets: list[Detection], cost: CostModel,
                      roi: RoiConfig | None = None, first_id: int = 0):
    """Measure every level of every detection on the real image.

    Returns (objects, encoded): the program's objects, and per object id the ROI and context
    JPEGs so a packet can be assembled for any assignment without re-reading the image.
    """
    roi = roi or RoiConfig()
    objs, encoded = [], {}
    for k, d in enumerate(dets):
        rec = SemanticRecord.from_detection(d, 0.0, 0, P1)
        n_rec = len(rec.pack())
        t0 = time.perf_counter()
        roi_b = extract_roi(img, rec, roi)
        t_roi = time.perf_counter() - t0
        t0 = time.perf_counter()
        ctx_b = extract_context(img, rec, roi)
        t_ctx = time.perf_counter() - t0
        sizes = {P0: 0.0, P1: float(n_rec), P2: float(n_rec + _BLOB.size + len(roi_b)),
                 P3: float(n_rec + 2 * _BLOB.size + len(roi_b) + len(ctx_b))}
        proc = {P0: 0.0, P1: 0.0, P2: t_roi, P3: t_roi + t_ctx}
        oid = first_id + k
        objs.append(make_object(oid, float(d.p), sizes, proc, cost))
        encoded[oid] = EncodedDetection(d, roi_b, ctx_b, t_roi, t_ctx)
    return objs, encoded


def assemble_packet(ids: list[int], assignment: dict, encoded: dict, image_id: int,
                    timestamp: float) -> SemanticPacket:
    """The packet a scene sends under an assignment (records P1+, ROI P2+, context P3)."""
    pkt = SemanticPacket(image_id, timestamp)
    for oid in ids:
        lv = assignment.get(oid, P0)
        if lv == P0:
            continue
        e = encoded[oid]
        pkt.records.append(SemanticRecord.from_detection(e.detection, timestamp, image_id, lv))
        i = len(pkt.records) - 1
        if lv >= P2:
            pkt.blobs.append((i, 1, e.roi))
        if lv >= P3:
            pkt.blobs.append((i, 2, e.context))
    return pkt


# ---------------------------------------------------------------- objective and solver
def _obj_J(opt: LevelOption, jw: JointWeights) -> float:
    return jw.alpha * opt.energy_J + jw.beta * opt.bytes / MB + jw.gamma * opt.latency_s / HOUR


def _opt(o: JointObject, level: int) -> LevelOption:
    for op in o.options:
        if op.level == level:
            return op
    raise KeyError(level)


@dataclass
class JointResult:
    assignment: dict
    J: float
    E_total_J: float
    D_tx_MB: float
    T_total_h: float
    accuracy: float
    info_preservation: float
    feasible: bool                # both constraints met
    reachable: bool               # the constraints can be met at all
    J_lower_bound: float          # unconstrained per-object optimum: a true lower bound
    gap: float = field(default=0.0)

    @property
    def valid(self) -> bool:
        """A solution can never beat the unconstrained bound (bracket check)."""
        return self.J + 1e-9 * max(1.0, abs(self.J_lower_bound)) >= self.J_lower_bound


def _totals(objs, assign, jw):
    chosen = [_opt(o, assign[o.id]) for o in objs]
    w = sum(o.weight for o in objs)
    imax = sum(o.info_max for o in objs)
    return (sum(_obj_J(c, jw) for c in chosen), sum(c.energy_J for c in chosen),
            sum(c.bytes for c in chosen) / MB, sum(c.latency_s for c in chosen) / HOUR,
            sum(c.acc for c in chosen) / w if w else 1.0,
            sum(c.info for c in chosen) / imax if imax else 1.0)


def solve(objs: list[JointObject], weights: JointWeights | None = None,
          constraints: JointConstraints | None = None) -> JointResult:
    """Unconstrained per-object optimum, then greedy repair toward both floors."""
    jw, con = weights or JointWeights(), constraints or JointConstraints()
    if not objs:
        return JointResult({}, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, True, True, 0.0, 0.0)

    level = {o.id: min(o.options, key=lambda op: _obj_J(op, jw)).level for o in objs}
    j_lower = sum(_obj_J(_opt(o, level[o.id]), jw) for o in objs)
    w_tot = sum(o.weight for o in objs)
    i_max = sum(o.info_max for o in objs)
    a_ceiling = (sum(o.weight for o in objs if any(op.level >= P1 for op in o.options)) / w_tot
                 if w_tot else 1.0)
    reachable = con.a_min <= a_ceiling + 1e-12 and con.i_min <= 1.0 + 1e-12 and i_max > 0

    by_id = {o.id: o for o in objs}
    a_sum = sum(_opt(o, level[o.id]).acc for o in objs)
    i_sum = sum(_opt(o, level[o.id]).info for o in objs)

    def apply(o, to):
        nonlocal a_sum, i_sum
        cur, new = _opt(o, level[o.id]), _opt(o, to)
        a_sum += new.acc - cur.acc
        i_sum += new.info - cur.info
        level[o.id] = to

    for o in objs:                                   # upgrades that cost nothing are always taken
        cur_j = _obj_J(_opt(o, level[o.id]), jw)
        for op in sorted(o.options, key=lambda z: z.level):
            if op.level > level[o.id] and _obj_J(op, jw) <= cur_j + 1e-15:
                apply(o, op.level)
                cur_j = _obj_J(op, jw)

    if w_tot and con.a_min > 0:                      # accuracy: cheapest objects to start reporting
        cand = []
        for o in objs:
            if _opt(o, level[o.id]).acc > 0:
                continue
            reach = [op for op in o.options if op.acc > 0]
            if reach:
                tgt = min(reach, key=lambda op: _obj_J(op, jw))
                cand.append((_obj_J(tgt, jw) - _obj_J(_opt(o, level[o.id]), jw), o.id, tgt.level))
        for _, oid, lv in sorted(cand):
            if a_sum / w_tot + 1e-12 >= con.a_min:
                break
            if level[oid] < lv:
                apply(by_id[oid], lv)

    if i_max and con.i_min > 0:                      # information: cheapest J per unit of info
        heap = []

        def push_next(o):
            nxt = next((op for op in sorted(o.options, key=lambda z: z.level)
                        if op.level > level[o.id]), None)
            if nxt is None:
                return
            cur = _opt(o, level[o.id])
            d_info = nxt.info - cur.info
            if d_info > 0:
                d_j = _obj_J(nxt, jw) - _obj_J(cur, jw)
                heapq.heappush(heap, (d_j / d_info, o.id, cur.level, nxt.level))

        for o in objs:
            push_next(o)
        while heap and i_sum / i_max + 1e-12 < con.i_min:
            _, oid, frm, to = heapq.heappop(heap)
            if level[oid] != frm:                    # stale entry
                push_next(by_id[oid])
                continue
            apply(by_id[oid], to)
            push_next(by_id[oid])

    j, e, d, t, a, i = _totals(objs, level, jw)
    feasible = a + 1e-9 >= con.a_min and i + 1e-9 >= con.i_min
    # relative gap; with a zero bound (everything at P0 is free) only "no gap" is meaningful
    gap = (j - j_lower) / j_lower if j_lower > 0 else (0.0 if j <= 1e-15 else float("inf"))
    return JointResult(dict(level), j, e, d, t, a, i, feasible, reachable, j_lower, gap)


PARETO_GRID = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, 10, 1), (10, 1, 1), (1, 1, 10))
I_MIN_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def pareto(objs: list[JointObject], constraints: JointConstraints | None = None,
           grid=PARETO_GRID) -> list[tuple[JointWeights, JointResult]]:
    """Solve at several (alpha, beta, gamma): the corners of the E / D / T trade-off.

    When E, D and T all grow with the bytes of a level (true for this pipeline), the weights
    rarely change the decision; the constraint sweep below is then the informative curve.
    """
    return [(JointWeights(*g), solve(objs, JointWeights(*g), constraints)) for g in grid]


def constraint_sweep(objs: list[JointObject], weights: JointWeights | None = None,
                     a_min: float = 0.9, i_mins=I_MIN_SWEEP) -> list[tuple[JointConstraints, JointResult]]:
    """Solve at increasing information floors: what each extra point of eq. (22) costs."""
    return [(JointConstraints(a_min, i), solve(objs, weights, JointConstraints(a_min, i)))
            for i in i_mins]
