"""Direct downlink and optional inter-satellite forwarding (paper V-D, VII).

    Primary -> Ground station                         (23)
    Primary -> Relay satellite -> Ground station      (24)
    J_direct = l_E E_direct + l_T T_direct            (25)
    J_relay  = l_E E_relay  + l_T T_relay             (26)
    E_comm   = min(E_direct, E_relay)  s.t. window and deadline constraints   (16)

The relay only forwards packets that the primary has already built: it never sees the image.

Every link is a ``LinkSchedule``: the time windows in which it exists (ground passes from the
SGP4 propagator in ``sat7.orbit``; ISL windows where the two satellites are in range and the
line between them clears the Earth) and a rate in each. Packets on one link go first-in
first-out and may span several windows; a link is busy while it transmits, so a large raw
image delays the packets behind it -- which is how B0's latency arises.

ASSUMPTIONS (in ``RelayConfig``): the relay's orbit, the ISL range and rate, the efficiency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from skyfield.api import load

from .energy import PowerConfig
from .orbit import (R_EARTH, GroundStation, LinkConfig, OrbitConfig, find_passes,
                    make_satellite)


@dataclass
class Window:
    start: float          # seconds from the experiment epoch
    end: float
    rate_bps: float

    @property
    def capacity_bytes(self) -> float:
        return (self.end - self.start) * self.rate_bps / 8.0


@dataclass
class Plan:
    start: float
    finish: float
    tx_s: float           # time actually spent transmitting (T_tx, eq. 13)
    rate_bps: float       # effective rate = 8 D / tx_s


class LinkSchedule:
    """One FIFO link made of time windows."""

    def __init__(self, windows: list[Window], name: str = ""):
        self.windows = sorted(windows, key=lambda w: w.start)
        self.busy_until = -math.inf
        self.name = name

    def plan(self, t_ready: float, nbytes: float) -> Plan | None:
        """When would ``nbytes`` ready at ``t_ready`` finish? None if beyond the last window."""
        t = max(t_ready, self.busy_until)
        left, start, tx = 8.0 * nbytes, None, 0.0
        for w in self.windows:
            if w.end <= t:
                continue
            s = max(t, w.start)
            avail = (w.end - s) * w.rate_bps
            if start is None:
                start = s
            if avail >= left:
                dt = left / w.rate_bps
                tx += dt
                return Plan(start, s + dt, tx, 8.0 * nbytes / tx if tx > 0 else w.rate_bps)
            left -= avail
            tx += w.end - s
            t = w.end
        return None

    def commit(self, p: Plan) -> None:
        self.busy_until = max(self.busy_until, p.finish)


def merge_windows(windows: list[Window]) -> list[Window]:
    """One radio: where windows overlap in time (two stations in view), keep the best rate.

    Splits the timeline at every window edge and, on each piece, keeps the highest rate among
    the windows that cover it. Capacity is never counted twice for the same second.
    """
    edges = sorted({t for w in windows for t in (w.start, w.end)})
    out: list[Window] = []
    for a, b in zip(edges, edges[1:]):
        rates = [w.rate_bps for w in windows if w.start <= a and w.end >= b]
        if not rates:
            continue
        r = max(rates)
        if out and out[-1].end == a and out[-1].rate_bps == r:
            out[-1].end = b                         # extend the previous piece
        else:
            out.append(Window(a, b, r))
    return out


def ground_windows(orbit: OrbitConfig, gs: GroundStation | list[GroundStation], link: LinkConfig,
                   start: datetime, hours: float, ts=None) -> list[Window]:
    """Ground passes of one satellite over one station or a network, each with its mean rate."""
    ts = ts or load.timescale()
    sat = make_satellite(orbit, ts)
    stations = gs if isinstance(gs, list) else [gs]
    out = []
    for g in stations:
        for p in find_passes(sat, g, start, hours, link, orbit.altitude_km, ts=ts):
            rate = 8.0 * p.capacity_bytes / p.duration_s
            out.append(Window((p.rise - start).total_seconds(), (p.set - start).total_seconds(),
                              rate))
    return merge_windows(out) if len(stations) > 1 else sorted(out, key=lambda w: w.start)


@dataclass
class RelayConfig:
    """ASSUMPTIONS for the optional relay. A second LEO in another plane (RAAN offset)."""
    raan_offset_deg: float = 60.0
    isl_max_range_km: float = 4500.0   # optical terminal pointing / link-budget limit
    isl_rate_bps: float = 100e6        # optical ISL data rate
    isl_efficiency: float = 0.8
    grazing_alt_km: float = 100.0      # the line of sight must clear the atmosphere
    step_s: float = 20.0


def _segment_clears_earth(a: np.ndarray, b: np.ndarray, r_min: float) -> np.ndarray:
    """Vectorised: does the segment a->b stay above radius r_min (km)?"""
    d = b - a
    dd = np.einsum("ij,ij->i", d, d)
    t = np.clip(-np.einsum("ij,ij->i", a, d) / np.where(dd > 0, dd, 1), 0.0, 1.0)
    closest = a + t[:, None] * d
    return np.linalg.norm(closest, axis=1) > r_min


def isl_windows(primary: OrbitConfig, relay: OrbitConfig, start: datetime, hours: float,
                cfg: RelayConfig, ts=None) -> list[Window]:
    ts = ts or load.timescale()
    s1 = make_satellite(primary, ts)
    s2 = make_satellite(relay, ts)
    secs = np.arange(0.0, hours * 3600.0 + cfg.step_s, cfg.step_s)
    t = ts.from_datetimes([start + timedelta(seconds=float(x)) for x in secs])
    a, b = s1.at(t).position.km.T, s2.at(t).position.km.T
    ok = (np.linalg.norm(a - b, axis=1) <= cfg.isl_max_range_km) & \
        _segment_clears_earth(a, b, R_EARTH + cfg.grazing_alt_km)
    rate = cfg.isl_rate_bps * cfg.isl_efficiency
    out, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j > i:
                out.append(Window(float(secs[i]), float(secs[j]), rate))
            i = j + 1
        else:
            i += 1
    return out


def relay_orbit(primary: OrbitConfig, cfg: RelayConfig) -> OrbitConfig:
    return OrbitConfig(primary.altitude_km, primary.inclination_deg,
                       (primary.raan_deg + cfg.raan_offset_deg) % 360.0, primary.epoch)


@dataclass
class PathChoice:
    path: str                     # "direct" | "relay" | "none"
    finish: float = math.inf
    e_comm: float = math.inf      # J
    t_comm: float = math.inf      # s, from packet ready to delivered (eq. 29 for relay)
    e_isl: float = 0.0
    e_gs: float = 0.0
    options: dict = field(default_factory=dict)   # (E, T) of every option, for eq. (35)


@dataclass
class PathPolicy:
    allow_relay: bool = True
    lambda_e: float = 1.0          # eq. (25)/(26) weight on energy, 1/J
    lambda_t: float = 1.0 / 60.0   # weight on latency, 1/s  (1 J is worth 1 minute)
    deadline_s: float = 6 * 3600.0


class Network:
    """Primary downlink, optional ISL + relay downlink, and the path decision."""

    def __init__(self, direct: list[Window], isl: list[Window] | None = None,
                 relay_gs: list[Window] | None = None, power: PowerConfig | None = None,
                 policy: PathPolicy | None = None):
        self.direct = LinkSchedule(direct, "direct")
        self.isl = LinkSchedule(isl or [], "isl")
        self.relay_gs = LinkSchedule(relay_gs or [], "relay_gs")
        self.power = power or PowerConfig()
        self.policy = policy or PathPolicy()

    def _options(self, t_ready: float, nbytes: float):
        opts = {}
        d = self.direct.plan(t_ready, nbytes)
        if d is not None:                              # eqs. (12)-(14)
            opts["direct"] = (self.power.p_tx * d.tx_s, d.finish - t_ready, (d,), 0.0, 0.0)
        if self.policy.allow_relay:
            i = self.isl.plan(t_ready, nbytes)
            if i is not None:
                g = self.relay_gs.plan(i.finish, nbytes)
                if g is not None:                      # eqs. (15), (29)
                    e_isl, e_gs = self.power.p_isl * i.tx_s, self.power.p_relay_gs * g.tx_s
                    opts["relay"] = (e_isl + e_gs, g.finish - t_ready, (i, g), e_isl, e_gs)
        return opts

    def send(self, t_ready: float, nbytes: float) -> PathChoice:
        opts = self._options(t_ready, nbytes)
        pol = self.policy
        on_time = {k: v for k, v in opts.items() if v[1] <= pol.deadline_s}
        pool = on_time or opts                          # nothing on time: least late wins
        if not pool:
            return PathChoice("none", options={k: (v[0], v[1]) for k, v in opts.items()})
        if on_time:                                     # eqs. (25), (26)
            key = min(pool, key=lambda k: pol.lambda_e * pool[k][0] + pol.lambda_t * pool[k][1])
        else:
            key = min(pool, key=lambda k: pool[k][1])
        e, t, plans, e_isl, e_gs = pool[key]
        if key == "direct":
            self.direct.commit(plans[0])
        else:
            self.isl.commit(plans[0])
            self.relay_gs.commit(plans[1])
        return PathChoice(key, t_ready + t, e, t, e_isl, e_gs,
                          {k: (v[0], v[1]) for k, v in opts.items()})
