"""The dashboard: its loaders read what the scripts wrote, and every page renders without error.

The runner (dashboard/runner.py) is checked against the script's own argparse, so the
Simulation page cannot pass a flag the script does not know or show a default it does not use.
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

import runner  # noqa: E402
from loaders import (RESULTS, ablation_deltas, available_sources, compare_campaigns,  # noqa: E402
                     live_sources, load_measured, load_paper)

needs_sim = pytest.mark.skipif(not (RESULTS / "paper_b0_b4_sim.json").exists(),
                               reason="run scripts/paper_b0_b4.py first")


@needs_sim
def test_loader_reads_every_paper_output():
    assert "SIM" in available_sources()
    r = load_paper("SIM")
    assert r.label == "SIM"
    assert list(r.campaign.config) == ["B0", "B1", "B2", "B3", "B4", "B4+J"]
    assert r.joint is not None and set(r.joint.sweep) == {"weights", "i_min"}
    assert {"direct", "isl", "relay_gs"} >= set(r.links.link)
    assert len(r.packets) == r.meta["scenes"]
    assert r.scene_image.exists() and (r.scene.kind == "detection").any()


@needs_sim
def test_ablation_deltas_are_relative_to_full_b4():
    d = ablation_deltas(load_paper("SIM").ablations)
    assert len(d) == 6 and "B4 complet" not in set(d.ablation)
    no_relay = d[d.ablation == "sans relais"].iloc[0]
    assert no_relay["octets (%)"] == pytest.approx(0, abs=1e-9)   # same packets, other path


def test_measured_loader():
    m = load_measured()
    if m is None:
        pytest.skip("WP results absent")
    assert len(m.strategies) == 6 and m.detector["test"]["mAP50"] > 0


@needs_sim
@pytest.mark.parametrize("page", ["Vue d'ensemble", "Communication", "Compromis", "Scène",
                                  "Mesures réelles (fusion)", "Simulation", "Comparer",
                                  "Ablations", "Résultats mesurés (WP)", "Hypothèses"])
def test_every_page_renders(page):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio(key="page").set_value(page)
    at.sidebar.radio(key="source").set_value("SIM")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.header[0].value


# ------------------------------------------------------------------ the Simulation page's runner
def script_parser() -> argparse.ArgumentParser:
    """The ArgumentParser that scripts/paper_b0_b4.py builds, caught before it parses sys.argv."""
    spec = importlib.util.spec_from_file_location("paper_b0_b4_under_test", runner.SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Caught(Exception):
        pass

    def catch(self, *a, **k):
        raise Caught(self)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", catch):
        try:
            mod.main()
        except Caught as c:
            return c.args[0]
    raise AssertionError("paper_b0_b4.main() never parsed its arguments")


@pytest.fixture(scope="module")
def parser():
    return script_parser()


def test_runner_defaults_are_the_scripts_defaults(parser):
    defaults = vars(parser.parse_args([]))
    assert {k: defaults[k] for k in runner.DEFAULTS} == runner.DEFAULTS


def test_runner_command_line_round_trips_through_the_script(parser, tmp_path):
    p = runner.normalise({"scenes": 6, "tile": 768, "overlap": 0.1, "fusion": "nms", "conf": 0.35,
                          "stations": "sfax,svalbard", "deadline_hours": 3.5,
                          "weights_abg": "2,1,0.5", "a_min": 0.8, "i_min": 0.3, "seed": 7})
    cmd = runner.command(p, tmp_path / "run")
    assert cmd[0] == str(runner.PYTHON["classic"]) and cmd[2] == str(runner.SCRIPT)
    ns = vars(parser.parse_args(cmd[3:]))
    assert {k: ns[k] for k in p} == p
    assert ns["out"] == tmp_path / "run" and ns["tag"] == ""     # isolated dir, no suffix
    yolo = runner.normalise({"detector": "yolo", "source": "swaths"})
    assert runner.command(yolo, tmp_path)[0] == str(runner.PYTHON["yolo"])
    assert runner.expected_label(yolo) == "REAL" and runner.expected_label(p) == "SIM"
    assert runner.expected_label(runner.normalise({"source": "swaths"})) == "SIM"   # classic


def test_runner_hash_ignores_key_order_but_not_values():
    a = runner.normalise({"scenes": 6, "conf": 0.25})
    b = runner.normalise(dict(reversed(list(a.items()))))
    assert runner.params_hash(a) == runner.params_hash(b)
    assert runner.params_hash(a) != runner.params_hash({**a, "seed": 1})


def test_preflight_names_what_is_missing(tmp_path):
    with (mock.patch.dict(runner.PYTHON, {"yolo": tmp_path / "none" / "python.exe"}),
          mock.patch.object(runner, "WEIGHTS", tmp_path / "best.pt"),
          mock.patch.object(runner, "AIRBUS_TEST", tmp_path / "test")):
        msgs = runner.preflight(runner.normalise({"detector": "yolo", "source": "swaths"}))
        assert len(msgs) == 3 and any(".venv312" in m for m in msgs)
        assert runner.preflight(runner.normalise({})) == []          # synthetic + classic


def test_progress_follows_the_scripts_output():
    done, _ = runner.progress("  synthetic_000: 2048x2048,  20 ships, direct  17 / SAHI ...", 0, 2)
    assert done == pytest.approx(0.3)
    done, msg = runner.progress("eq. (20) on 241 objects: J 0.22", done, 2)
    assert done == 0.65 and msg
    assert runner.progress("Saved paper_b0_b4.csv|.json in x", done, 2)[0] == 1.0
    assert runner.progress("random line", 0.4, 2) == (0.4, None)


def test_failure_reason_is_the_last_line():
    log = "a\n\nsys.exit: no test images under x\n"
    assert runner.failure_reason(log) == "sys.exit: no test images under x"
    assert "torch" in runner.failure_reason("Traceback\nModuleNotFoundError: No module named 'torch'")


def test_a_live_run_writes_only_its_own_directory(tmp_path):
    """One tiny real run: outputs land in results/live/<id>/, the loaders read them back."""
    committed = {f: f.stat().st_mtime_ns for f in RESULTS.glob("paper_*")}
    p = {"scenes": 1, "scene_px": 1024}
    job = runner.start(p, live=tmp_path / "live", now=datetime(2026, 10, 9, 14, 30, 5))
    run = runner.finish(job)
    assert run.info["returncode"] == 0, run.log[-2000:]
    assert run.id == f"20261009-143005_{runner.params_hash(runner.normalise(p))}"
    assert json.loads((run.dir / "params.json").read_text()) == runner.normalise(p)
    assert {"paper_b0_b4.json", "paper_b0_b4.csv", "paper_packets.csv", "paper_scene.jpg",
            "run.log", "run.json"} <= {f.name for f in run.dir.iterdir()}
    assert {f: f.stat().st_mtime_ns for f in RESULTS.glob("paper_*")} == committed
    assert live_sources(tmp_path) == [f"live:{run.id}"]
    r = load_paper(f"live:{run.id}", results=tmp_path)
    assert r.label == "SIM" and r.live_id == run.id and r.meta["scenes"] == 1
    assert runner.find_cached(p, live=tmp_path / "live").id == run.id
    assert runner.find_cached({**p, "seed": 3}, live=tmp_path / "live") is None
    assert [x.id for x in runner.live_runs(tmp_path / "live")] == [run.id]


@needs_sim
def test_compare_puts_two_runs_side_by_side():
    sim = load_paper("SIM").campaign
    t = compare_campaigns(sim, sim.assign(recall=sim.recall + 0.1), ["recall", "D_tx_MB"])
    assert list(t.config) == list(sim.config)
    assert list(t.columns[1:]) == ["rappel · A", "rappel · B", "rappel · Δ",
                                   "Mo envoyés · A", "Mo envoyés · B", "Mo envoyés · Δ"]
    assert t["rappel · Δ"].to_numpy() == pytest.approx(0.1)
    assert (t["Mo envoyés · Δ"] == 0).all()
