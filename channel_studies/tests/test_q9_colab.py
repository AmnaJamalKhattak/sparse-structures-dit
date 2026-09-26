"""No pretrained weights or Colab account needed to verify the form workflow."""

# Execute only repository-owned notebook cells with model execution mocked out.
# ruff: noqa: S102

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.experiments.q9_colab import (
    build_config,
    run_budget,
    run_experiment,
    run_logged,
    save_config,
)
from src.experiments.text_image_coupling import preset_config


@pytest.mark.parametrize("model", ["flux1-dev", "flux-schnell"])
@pytest.mark.parametrize("mode", ["smoke", "discovery", "screen", "confirm"])
def test_default_forms_preserve_presets(model, mode):
    assert asdict(build_config(model, mode, workload="full")) == asdict(preset_config(model, mode))


def test_form_overrides_need_no_python_and_stay_local():
    cfg = build_config(
        "flux-schnell",
        "screen",
        "512",
        vram_gib=16,
        bf16=False,
        advanced={
            "sites": "-1, 17",
            "steps": "0,3",
            "seeds": "8,9",
            "prompts": "a blue cup || a red cup",
            "methods": "zero",
            "prompt_count": 1,
        },
    )
    assert cfg.prompts == ["a blue cup"] and cfg.steps == [0, 3]
    assert cfg.offload and cfg.dtype == "fp16" and cfg.resolution == 512
    assert cfg.output_dir.startswith("/content/q9_work/")


@pytest.mark.parametrize(
    "advanced",
    [
        {"sites": "39"},
        {"steps": "999"},
        {"prompt_count": 999},
        {"methods": "typo"},
        {"seeds": "-1"},
        {"output_dir": "/content/drive"},
        {"prompts": "same", "calibration_prompts": "same"},
    ],
)
def test_invalid_form_settings_fail_before_model_load(advanced):
    with pytest.raises(ValueError):
        build_config(advanced=advanced)


def test_budget_distinguishes_discovery_screen_and_confirm():
    discovery = run_budget(build_config(mode="discovery", workload="full"))
    assert discovery["calibration_trajectories_if_uncached"] == 24
    assert discovery["evaluation_pairs"] == discovery["edited_single_forwards"] == 0
    confirm = run_budget(build_config(mode="confirm", workload="full"))
    assert confirm["edited_full_trajectories"] == 576
    assert confirm["clean_evaluation_trajectories"] == 75
    screen = run_budget(build_config(mode="screen"))
    assert screen["edited_single_forwards"] > 0 and screen["edited_full_trajectories"] == 0


@pytest.mark.parametrize("model", ["flux1-dev", "flux-schnell"])
def test_pilot_limits_work_and_preserves_pairing_and_calibration_split(model):
    cfg = build_config(model)
    budget = run_budget(cfg)
    assert len(cfg.prompts) == len(cfg.calibration_prompts) == 3
    assert len(cfg.seeds) == len(cfg.calibration_seeds) == 1
    assert not set(cfg.prompts) & set(cfg.calibration_prompts)
    assert cfg.sites == [17] and cfg.steps == [0]
    assert cfg.methods == ["remove_direction", "norm_matched", "zero", "ordinary_zero"]
    assert cfg.rescues == ["none"] and not cfg.include_empty
    assert budget["edited_single_forwards"] == 12
    assert budget["clean_evaluation_trajectories"] == 3
    assert budget["calibration_trajectories_if_uncached"] == 3
    confirm = build_config(model, "confirm")
    assert not set(cfg.prompts) & set(confirm.prompts)
    assert cfg.calibration_identity() == confirm.calibration_identity()
    assert run_budget(confirm)["edited_full_trajectories"] == 12
    assert run_budget(build_config(model, "discovery"))["calibration_trajectories_if_uncached"] == 3


def test_optional_empty_baseline_can_be_restored_without_changing_preset_default():
    assert not build_config(advanced={"include_empty": "preset"}).include_empty
    assert build_config(advanced={"include_empty": "yes"}).include_empty
    assert build_config(workload="full", advanced={"include_empty": "preset"}).include_empty


def test_main_notebook_has_one_simple_q9_workflow_and_no_standalone():
    assert not Path("Q9_Colab.ipynb").exists()
    notebook = json.loads(Path("Figure3_Colab.ipynb").read_text(encoding="utf-8"))
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert [cell_id for cell_id in ids if str(cell_id).startswith("q9_")] == [
        "q9_intro",
        "q9_config",
        "q9_run",
    ]
    cells = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("id")
    }
    assert "# @param ['flux-schnell', 'flux1-dev']" in cells["q9_config"]
    assert "# @param ['screen', 'confirm']" in cells["q9_config"]
    assert "workload='pilot'" in cells["q9_run"]
    assert "export_compact" in cells["q9_run"] and "if USE_DRIVE" in cells["q9_run"]
    assert "advanced" not in cells["q9_run"] and "Q9_CONFIRM_FULL_RUN" not in cells["q9_run"]
    assert ids.index("q9_config") < ids.index("q9_run") < ids.index("XcA1FvnJhZlY")
    for cell_id in ("q9_config", "q9_run"):
        compile(cells[cell_id], cell_id, "exec")


def test_generated_config_stays_outside_drive(tmp_path):
    cfg = build_config("flux-schnell", "screen")
    path = Path(save_config(cfg, tmp_path))
    assert path.parent == tmp_path
    assert json.loads(path.read_text()) == asdict(cfg)


def test_notebook_runs_and_exports_automatically(tmp_path, monkeypatch):
    from types import SimpleNamespace

    nb = json.loads(Path("Figure3_Colab.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c.get("id")}
    commands = []
    exports = []
    monkeypatch.setattr("src.experiments.q9_colab.show_results", lambda cfg: "shown")
    monkeypatch.setattr(
        "src.experiments.q9_colab.run_experiment", lambda path, **kwargs: commands.append(path)
    )
    monkeypatch.setattr("src.experiments.q9_colab.save_config", lambda cfg: str(tmp_path / "q9.json"))
    monkeypatch.setattr(
        "src.experiments.q9_report.export_compact",
        lambda cfg, destination: exports.append(destination),
    )
    namespace = {
        "REPO_DIR": "/content/channel_studies",
        "USE_DRIVE": True,
        "DRIVE_ROOT": "/content/drive/MyDrive/Research/MA",
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda _: SimpleNamespace(total_memory=40 * 2**30),
                is_bf16_supported=lambda: True,
            )
        ),
    }
    exec(cells["q9_config"], namespace)
    exec(cells["q9_run"], namespace)
    assert len(commands) == 1 and namespace["q9_finished"]
    assert commands[0] == namespace["Q9_CONFIG_PATH"]
    assert namespace["q9_result"] == "shown"
    assert exports == [
        "/content/drive/MyDrive/Research/MA/flux-schnell/q9_compact/screen"
    ]


def test_notebook_failed_run_stays_unfinished(monkeypatch):
    from types import SimpleNamespace

    nb = json.loads(Path("Figure3_Colab.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c.get("id")}
    monkeypatch.setattr("src.experiments.q9_colab.save_config", lambda cfg: "/tmp/q9.json")

    def fail(*_args, **_kwargs):
        raise RuntimeError("model failure")

    monkeypatch.setattr("src.experiments.q9_colab.run_experiment", fail)
    namespace = {
        "REPO_DIR": "/content/channel_studies",
        "USE_DRIVE": False,
        "DRIVE_ROOT": "/content/local",
        "Q9_MODEL": "flux-schnell",
        "Q9_STAGE": "screen",
        "q9_finished": True,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda _: SimpleNamespace(total_memory=40 * 2**30),
                is_bf16_supported=lambda: True,
            )
        ),
    }
    with pytest.raises(RuntimeError, match="model failure"):
        exec(cells["q9_run"], namespace)
    assert not namespace["q9_finished"]


@pytest.mark.parametrize("exit_code", [0, 1])
def test_child_output_and_failure_are_visible_and_logged(tmp_path, capsys, exit_code):
    import sys

    log = tmp_path / "q9.log"
    cmd = [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys; print('child progress'); print('underlying diagnostic', file=sys.stderr); "
            f"sys.exit({exit_code})"
        ),
    ]
    if exit_code:
        with pytest.raises(RuntimeError, match="underlying diagnostic"):
            run_logged(cmd, log)
    else:
        assert run_logged(cmd, log) == log
    output = capsys.readouterr().out
    assert "child progress" in output and "underlying diagnostic" in output
    assert "underlying diagnostic" in log.read_text()


def test_experiment_wrapper_uses_current_interpreter_and_config(tmp_path, monkeypatch):
    import sys

    captured = []

    def capture(command, log_path, cwd):
        captured.append((command, log_path, cwd))
        return log_path

    monkeypatch.setattr("src.experiments.q9_colab.run_logged", capture)
    cfg = tmp_path / "config.json"
    first = run_experiment(cfg, cwd=tmp_path)
    second = run_experiment(cfg, cwd=tmp_path)
    command, _, cwd = captured[0]
    assert command == [
        sys.executable,
        "-u",
        "-m",
        "src.experiments.text_image_coupling",
        "--config",
        str(cfg.resolve()),
    ]
    assert cwd == tmp_path and first.parent == tmp_path / "q9_logs"
    assert first != second
