"""Launch scripts/paper_b0_b4.py from the dashboard, one isolated directory per run.

The dashboard still computes nothing: this module only builds the command line, starts the
script as a subprocess and finds its outputs afterwards. Every run lives in

    results/live/<YYYYmmdd-HHMMSS>_<hash of the parameters>/
        params.json   the parameters, exactly as passed to the script
        run.log       the script's stdout + stderr, written live
        run.json      return code, duration, interpreter (written when the run ends)
        paper_*       the script's usual outputs, with no suffix (--tag "")

so a live run can never overwrite the committed paper_*_sim.* files in results/.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # code/
SCRIPT = ROOT / "scripts" / "paper_b0_b4.py"
LIVE = ROOT / "results" / "live"
PYTHON = {"classic": ROOT / ".venv" / "Scripts" / "python.exe",     # Python 3.14, no torch
          "yolo": ROOT / ".venv312" / "Scripts" / "python.exe"}     # Python 3.12, torch + CUDA
WEIGHTS = ROOT / "runs" / "ships" / "weights" / "best.pt"
AIRBUS_TEST = ROOT / "data" / "yolo_ships" / "images" / "test"

# The script's own defaults for every flag the dashboard exposes (tests/test_dashboard.py
# checks them against its argparse, so they cannot drift apart).
DEFAULTS = {"source": "synthetic", "detector": "classic", "scenes": 24, "scene_px": 2048,
            "tile": 512, "overlap": 0.20, "fusion": "nmm", "conf": 0.25,
            "link": "cubesat_sband", "stations": "sfax", "deadline_hours": 6.0,
            "weights_abg": "1,1,1", "a_min": 0.90, "i_min": 0.50, "seed": 0}
SOURCES = ("synthetic", "swaths", "airbus")         # dota / coco need paths: command line only

# Measured on the dev laptop (CPU, classic detector, 2048 px synthetic scenes, 512 px tiles):
# 2 / 6 / 12 scenes took 6 / 21 / 36 s, i.e. a small fixed cost (orbit passes, ISL geometry)
# plus ~3 s per scene (detection, every configuration, eq. 20 sweeps).
EST_FIXED_S, EST_PER_SCENE_S = 2.0, 3.0

_SCENE_LINE = re.compile(r"^\s+\S+: \d+x\d+,")


def expected_label(p: dict) -> str:
    """The label the script will write: REAL needs real scenes AND the YOLO detector."""
    return "REAL" if p["source"] != "synthetic" and p["detector"] == "yolo" else "SIM"


def params_hash(p: dict) -> str:
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:8]


def normalise(p: dict) -> dict:
    """Every exposed flag present, typed like the script's argparse would type it."""
    q = {**DEFAULTS, **p}
    for k in ("scenes", "scene_px", "tile", "seed"):
        q[k] = int(q[k])
    for k in ("overlap", "conf", "deadline_hours", "a_min", "i_min"):
        q[k] = float(q[k])
    return q


def estimate_seconds(p: dict) -> float | None:
    """Rough runtime for the classic detector on synthetic scenes; None when it is unknowable."""
    if p["detector"] != "classic" or p["source"] != "synthetic":
        return None
    per = EST_PER_SCENE_S * (p["scene_px"] / 2048) ** 2 * (512 / p["tile"]) ** 2
    return EST_FIXED_S + per * p["scenes"]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def preflight(p: dict) -> list[str]:
    """What is missing on this machine for these parameters, in plain French, before launching."""
    problems = []
    py = PYTHON[p["detector"]]
    if not py.exists():
        env = ".venv312 (Python 3.12 + torch + CUDA)" if p["detector"] == "yolo" else ".venv"
        problems.append(f"Interpréteur absent : {_rel(py)} — créer l'environnement {env}.")
    if p["detector"] == "yolo" and not WEIGHTS.exists():
        problems.append(f"Poids YOLO absents : {_rel(WEIGHTS)} — entraîner avec "
                        "scripts/wp1_train_detector.py ou copier best.pt depuis la machine GPU.")
    if p["source"] in ("swaths", "airbus") and not AIRBUS_TEST.is_dir():
        problems.append(f"Tuiles Airbus absentes : {_rel(AIRBUS_TEST)} — les construire "
                        "avec scripts/wp0_build_dataset.py.")
    return problems


def command(p: dict, out_dir: Path) -> list[str]:
    """The exact command line: paper_b0_b4.py with every exposed flag, writing into out_dir."""
    args = [str(PYTHON[p["detector"]]), "-u", str(SCRIPT)]
    for k in DEFAULTS:
        args += [f"--{k.replace('_', '-')}", str(p[k])]
    return args + ["--out", str(out_dir), "--tag", ""]


@dataclass
class LiveRun:
    """One directory under results/live/."""
    id: str
    dir: Path
    params: dict
    info: dict | None              # run.json, None while running or if the UI was interrupted

    @property
    def complete(self) -> bool:
        return (self.dir / "paper_b0_b4.json").exists()

    @property
    def label(self) -> str:
        if self.complete:
            return json.loads((self.dir / "paper_b0_b4.json").read_text())["label"]
        return expected_label(self.params)

    @property
    def started(self) -> datetime:
        return datetime.strptime(self.id.split("_")[0], "%Y%m%d-%H%M%S")

    @property
    def log(self) -> str:
        f = self.dir / "run.log"
        return f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""


def live_runs(live: Path = LIVE) -> list[LiveRun]:
    """Every live run, most recent first."""
    if not live.is_dir():
        return []
    runs = []
    for d in sorted(live.iterdir(), reverse=True):
        if not (d / "params.json").exists():
            continue
        info = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else None
        runs.append(LiveRun(d.name, d, json.loads((d / "params.json").read_text()), info))
    return runs


def find_cached(p: dict, live: Path = LIVE) -> LiveRun | None:
    """The most recent complete run with exactly these parameters."""
    h = params_hash(normalise(p))
    return next((r for r in live_runs(live) if r.id.endswith(h) and r.complete), None)


def progress(line: str, done: float, n_scenes: int) -> tuple[float, str | None]:
    """Advance the progress bar from one line of the script's output (in its print order)."""
    if _SCENE_LINE.match(line):
        return min(done + 0.6 / max(n_scenes, 1), 0.6), f"Scène {line.split(':')[0].strip()} traitée"
    for marker, at, text in (("eq. (20) on", 0.65, "Transmission, ablations et balayages"),
                             ("=== B0 -> B4", 0.95, "Écriture des résultats"),
                             ("Saved ", 1.0, "Terminé")):
        if line.startswith(marker):
            return max(done, at), text
    return done, None


@dataclass
class Job:
    """A launched run: its directory and the process writing into it."""
    run: LiveRun
    proc: subprocess.Popen
    log: object                    # the open run.log handle, closed by finish()
    t0: float


def start(p: dict, live: Path = LIVE, now: datetime | None = None) -> Job:
    """Create the run directory and launch the script; its output goes straight to run.log.

    Writing to a file rather than a pipe means the run finishes even if the page is
    reloaded mid-run: the dashboard only tails the log.
    """
    p = normalise(p)
    run_id = f"{(now or datetime.now()).strftime('%Y%m%d-%H%M%S')}_{params_hash(p)}"
    out = live / run_id
    out.mkdir(parents=True)
    (out / "params.json").write_text(json.dumps(p, indent=2))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    log = open(out / "run.log", "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(command(p, out), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=env)
    except OSError:
        log.close()
        raise
    return Job(LiveRun(run_id, out, p, None), proc, log, time.perf_counter())


def finish(job: Job) -> LiveRun:
    """Wait for the process and record how the run ended in run.json."""
    rc = job.proc.wait()
    job.log.close()
    run = job.run
    run.info = {"returncode": rc, "seconds": round(time.perf_counter() - job.t0, 1),
                "python": _rel(PYTHON[run.params["detector"]]),
                "complete": run.complete}
    (run.dir / "run.json").write_text(json.dumps(run.info, indent=2))
    return run


def failure_reason(log: str) -> str:
    """The last meaningful line of a failed run's log (the script's sys.exit message or the error)."""
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    if not lines:
        return "le script s'est arrêté sans rien écrire"
    for hint, text in (("CUDA", "Pas de GPU CUDA utilisable sur cette machine."),
                       ("No module named 'torch'", "torch absent de cet environnement."),
                       ("No module named 'ultralytics'", "ultralytics absent de cet environnement.")):
        if any(hint in ln for ln in lines[-15:]):
            return f"{text} Dernière ligne : {lines[-1]}"
    return lines[-1]
