"""Tests for `train.sweep_arena` — A3-013's acceptance criteria.

The sweep's job is to produce a defensible table, so what is worth pinning is the reasoning around
the training rather than the training itself: that exactly one axis moves per run, that the
baseline is not trained twice, that the tail-of-run scoring reads every worker's episodes in the
order they finished, that ranking puts phase progression above reward, and that a config which
bought reward without phase progression is flagged without being silently dropped.

The one end-to-end test trains for a few thousand timesteps on one env, which is enough to prove
the stages wire together and the artefacts land; the real 100k-per-run sweep lives in the CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from common.config import ArenaTrainConfig
from train import sweep_arena
from train.sweep_arena import Run, SweepConfig


@pytest.fixture
def baseline() -> ArenaTrainConfig:
    return ArenaTrainConfig.from_yaml("arena")


@pytest.fixture
def sweep() -> SweepConfig:
    return SweepConfig.from_yaml("arena")


# --- config ------------------------------------------------------------------------------------


def test_sweep_section_loads_and_names_only_overridable_axes(sweep: SweepConfig) -> None:
    """Every axis in config must map to a real `train_arena` flag, or its run would be a no-op."""
    assert sweep.axes
    assert set(sweep.axes) <= set(sweep_arena.AXIS_FLAGS)
    assert sweep.budget_timesteps > 0
    assert len(sweep.confirm_seeds) >= 3, "a one-seed win is not a result"


def test_unknown_axis_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd axis name must fail loudly rather than silently training the baseline N times."""
    monkeypatch.setattr(
        sweep_arena, "load_yaml",
        lambda name: {"sweep": {
            "budget_timesteps": 1000, "control_style": "direct",
            "axes": {"learninng_rate": [0.1]}, "confirm_seeds": [0, 1, 2],
            "confirm_top_k": 2, "summary_episodes": 10, "reward_hack_phase_tolerance": 0.05,
        }},
    )
    with pytest.raises(ValueError, match="learninng_rate"):
        SweepConfig.from_yaml("arena")


# --- planning ----------------------------------------------------------------------------------


def test_explore_plan_varies_exactly_one_axis_per_run(
    baseline: ArenaTrainConfig, sweep: SweepConfig
) -> None:
    runs = sweep_arena.plan_explore(baseline, sweep)
    for run in runs[1:]:
        assert len(run.overrides) == 1, f"{run.name} moves more than one axis"
        assert run.axis in sweep.axes


def test_explore_plan_runs_the_baseline_once(
    baseline: ArenaTrainConfig, sweep: SweepConfig
) -> None:
    """The baseline appears in several axis lists; it must still be trained exactly once."""
    runs = sweep_arena.plan_explore(baseline, sweep)
    assert sum(run.is_baseline for run in runs) == 1
    assert runs[0].is_baseline
    for run in runs[1:]:
        assert not sweep_arena._same_value(run.value, getattr(baseline, run.axis))
    assert len({run.name for run in runs}) == len(runs)


def test_confirm_plan_covers_every_seed_but_the_one_already_run(sweep: SweepConfig) -> None:
    winner = Run(name="gamma-0-95_s0", stage="explore", axis="gamma", value=0.95,
                 seed=0, timesteps=1000, overrides={"gamma": 0.95})
    runs = sweep_arena.plan_confirm([winner], sweep, baseline_seed=0)
    assert [run.seed for run in runs] == [s for s in sweep.confirm_seeds if s != 0]
    assert all(run.overrides == {"gamma": 0.95} for run in runs)
    assert all(run.stage == "confirm" for run in runs)


def test_command_passes_the_axis_flag_and_keeps_models_out_of_the_repo(
    sweep: SweepConfig, tmp_path: Path
) -> None:
    run = Run(name="net_arch-128x128_s0", stage="explore", axis="net_arch", value=[128, 128],
              seed=0, timesteps=1000, overrides={"net_arch": [128, 128]})
    command = sweep_arena.build_command(run, sweep, tmp_path / "logs", tmp_path / "models",
                                        "arena", "cpu")
    assert command[:3] == [sys.executable, "-m", "train.train_arena"]
    flag = command.index("--net-arch")
    assert command[flag + 1:flag + 3] == ["128", "128"]
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--timesteps") + 1] == "1000"
    assert str(sweep_arena.DEFAULT_MODELS_DIR) not in command


# --- scoring -----------------------------------------------------------------------------------


def _write_monitor(run_dir: Path, worker: int, rows: list[tuple[float, ...]]) -> None:
    """Write a Monitor CSV of `(r, l, t, phase, spawners, enemies, damage)` rows."""
    path = run_dir / "monitor" / f"worker_{worker}.monitor.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['#{"t_start": 0.0, "env_id": "None"}',
             "r,l,t,phase,spawners_destroyed,enemies_killed,damage_taken"]
    lines += [",".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_episodes_are_merged_across_workers_in_completion_order(tmp_path: Path) -> None:
    _write_monitor(tmp_path, 0, [(1.0, 10, 0.1, 0, 0, 0, 0), (3.0, 10, 0.3, 2, 2, 0, 0)])
    _write_monitor(tmp_path, 1, [(2.0, 10, 0.2, 1, 1, 0, 0)])
    episodes = sweep_arena.read_episodes(tmp_path)
    assert [episode["r"] for episode in episodes] == [1.0, 2.0, 3.0]


def test_summary_averages_only_the_tail_of_the_run(tmp_path: Path) -> None:
    """Early episodes must not drag the score down: a config is judged on where it ended up."""
    _write_monitor(tmp_path, 0, [(-10.0, 10, float(i), 0, 0, 0, 5) for i in range(50)]
                   + [(20.0, 30, float(50 + i), 2, 2, 4, 1) for i in range(10)])
    summary = sweep_arena.summarise(tmp_path, episodes_window=10)
    assert summary["ep_rew_mean"] == pytest.approx(20.0)
    assert summary["phase_reached"] == pytest.approx(2.0)
    assert summary["episodes"] == 60
    assert summary["episodes_averaged"] == 10


def test_summary_refuses_a_run_that_recorded_nothing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no Monitor episodes"):
        sweep_arena.summarise(tmp_path, episodes_window=10)


def test_ranking_prefers_phase_progression_over_reward() -> None:
    """The central claim of the table: more reward does not beat more phase progression."""
    rich = {"label": "rich", "metrics": {"phase_reached": 0.2, "ep_rew_mean": 90.0}}
    progressing = {"label": "progressing", "metrics": {"phase_reached": 1.4, "ep_rew_mean": 10.0}}
    assert sorted([rich, progressing], key=sweep_arena.rank_key)[0] is progressing


def test_reward_without_phase_progression_is_flagged() -> None:
    base = {"label": "baseline", "is_baseline": True,
            "metrics": {"phase_reached": 1.0, "ep_rew_mean": 10.0}}
    farmer = {"label": "ent_coef=0.0", "is_baseline": False,
              "metrics": {"phase_reached": 1.0, "ep_rew_mean": 40.0}}
    honest = {"label": "gamma=0.95", "is_baseline": False,
              "metrics": {"phase_reached": 1.6, "ep_rew_mean": 25.0}}
    results = [base, farmer, honest]
    sweep_arena.flag_reward_hacking(results, base, tolerance=0.05)
    assert farmer["reward_hacking_suspected"] is True
    assert honest["reward_hacking_suspected"] is False
    assert base["reward_hacking_suspected"] is False


def test_the_flag_annotates_rather_than_disqualifying() -> None:
    """A config that progressed more must keep its rank even when the blunt flag catches it.

    The tolerance sits at one seed-noise width, so it fires on real but sub-noise improvements too.
    Dropping flagged rows from the shortlist would therefore promote the baseline over a config
    that genuinely progressed further — which is how the first run of this sweep went wrong.
    """
    base = {"label": "baseline", "is_baseline": True, "reward_hacking_suspected": False,
            "metrics": {"phase_reached": 0.01, "ep_rew_mean": -13.0}}
    better = {"label": "learning_rate=0.001", "is_baseline": False,
              "metrics": {"phase_reached": 0.04, "ep_rew_mean": -10.2}}
    sweep_arena.flag_reward_hacking([base, better], base, tolerance=0.05)
    assert better["reward_hacking_suspected"] is True, "the blunt flag does catch it"
    ranked = sorted([base, better], key=sweep_arena.rank_key)
    assert ranked[0] is better, "but it still outranks the baseline it beat on phase"


def test_aggregate_reports_the_spread_across_seeds() -> None:
    runs = [
        {"name": "g_s0", "seed": 0, "overrides": {"gamma": 0.95}, "is_baseline": False,
         "metrics": {"phase_reached": 1.0, "ep_rew_mean": 10.0}},
        {"name": "g_s1", "seed": 1, "overrides": {"gamma": 0.95}, "is_baseline": False,
         "metrics": {"phase_reached": 2.0, "ep_rew_mean": 30.0}},
    ]
    entry = sweep_arena.aggregate("gamma=0.95", runs)
    assert entry["metrics"]["phase_reached"] == pytest.approx(1.5)
    assert entry["spread"]["phase_reached"] == pytest.approx(0.5)
    assert entry["seeds"] == [0, 1]


# --- reporting ---------------------------------------------------------------------------------


def test_report_marks_the_flagged_row_and_names_the_winner() -> None:
    report = {
        "generated": "2026-09-05T10:00:00", "git_commit": "abc1234",
        "sweep": {"budget_timesteps": 100000, "control_style": "direct",
                  "confirm_seeds": [0, 1, 2], "confirm_top_k": 2, "summary_episodes": 100,
                  "reward_hack_phase_tolerance": 0.05},
        "baseline_config": {"algorithm": "PPO", "gamma": 0.99},
        "explore": [
            {"label": "gamma=0.95", "is_baseline": False, "reward_hacking_suspected": False,
             "metrics": {"phase_reached": 1.6, "ep_rew_mean": 25.0}},
            {"label": "ent_coef=0.0", "is_baseline": False, "reward_hacking_suspected": True,
             "metrics": {"phase_reached": 1.0, "ep_rew_mean": 40.0}},
        ],
        "confirm": [{"label": "gamma=0.95", "seeds": [0, 1, 2],
                     "metrics": {"phase_reached": 1.5, "ep_rew_mean": 24.0},
                     "spread": {"phase_reached": 0.1, "ep_rew_mean": 2.0}}],
        "winner": {"label": "gamma=0.95", "seeds": [0, 1, 2],
                   "metrics": {"phase_reached": 1.5, "ep_rew_mean": 24.0}},
    }
    markdown = sweep_arena.render_report(report)
    assert "reward hacking" in markdown
    assert "`ent_coef=0.0`" in markdown
    assert "## Winner" in markdown and "`gamma=0.95`" in markdown
    assert "abc1234" in markdown


def test_model_path_in_the_report_is_repo_relative() -> None:
    """A committed table must not name the home directory of whoever ran the sweep."""
    inside = sweep_arena.REPO_ROOT / "models" / "ppo_direct.zip"
    assert sweep_arena._repo_relative(inside) == str(Path("models") / "ppo_direct.zip")


def test_report_only_rerenders_from_the_recorded_runs(tmp_path: Path) -> None:
    """The JSON is the record; the Markdown is a view of it, rebuildable without training."""
    report = {
        "generated": "2026-09-05T10:00:00", "git_commit": "abc1234",
        "sweep": {"budget_timesteps": 100000, "control_style": "direct",
                  "confirm_seeds": [0, 1, 2], "confirm_top_k": 2, "summary_episodes": 100,
                  "reward_hack_phase_tolerance": 0.02},
        "baseline_config": {"algorithm": "PPO", "n_steps": 2048},
        "explore": [{"label": "n_steps=512", "is_baseline": False,
                     "reward_hacking_suspected": False,
                     "metrics": {"phase_reached": 0.08, "ep_rew_mean": -7.86}}],
    }
    (tmp_path / "sweep_runs.json").write_text(json.dumps(report), encoding="utf-8")
    sweep_arena.rerender(tmp_path)
    markdown = (tmp_path / "sweep_table.md").read_text(encoding="utf-8")
    assert "`n_steps=512`" in markdown and "abc1234" in markdown


# --- end to end --------------------------------------------------------------------------------


@pytest.mark.slow
def test_sweep_runs_end_to_end_and_writes_both_artefacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A miniature sweep: one axis, two values, tiny budgets — proves the stages wire together."""
    monkeypatch.setattr(
        SweepConfig, "from_yaml",
        classmethod(lambda cls, name="arena", section="sweep": SweepConfig(
            budget_timesteps=2048, control_style="direct", axes={"gamma": [0.95, 0.99]},
            confirm_seeds=[0, 1], confirm_top_k=1, summary_episodes=5,
            reward_hack_phase_tolerance=0.05)),
    )
    baseline = ArenaTrainConfig.from_yaml("arena")
    monkeypatch.setattr(
        ArenaTrainConfig, "from_yaml",
        classmethod(lambda cls, name="arena", section="training": type(baseline)(
            **{**baseline.__dict__, "n_envs": 1, "n_steps": 512, "batch_size": 64,
               "total_timesteps": 2048})),
    )

    args = sweep_arena.parse_args([
        "--sweep-dir", str(tmp_path / "logs"),
        "--results-dir", str(tmp_path / "results"),
        "--models-dir", str(tmp_path / "models"),
    ])
    report = sweep_arena.run_sweep(args)

    assert (tmp_path / "results" / "sweep_table.md").exists()
    written = json.loads((tmp_path / "results" / "sweep_runs.json").read_text(encoding="utf-8"))
    assert len(written["explore"]) == 2, "baseline plus the one non-baseline gamma"
    assert written["winner"]["label"] in {"baseline", "gamma=0.95"}
    assert (tmp_path / "models" / "ppo_direct_sweep.zip").exists(), "final stage must save a model"
    assert report["final"]["timesteps"] == 2048


@pytest.mark.slow
def test_the_sweep_never_overwrites_the_canonical_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incumbent in `models/<algo>_<style>.zip` must survive a sweep untouched.

    A sweep compares configs against each other at a reduced budget; it does not show that its
    winner beats the model already shipped. Promoting on rank alone is how a tuning run replaces a
    good agent with a worse one, which is exactly what this sweep's first full run would have done.
    """
    monkeypatch.setattr(
        SweepConfig, "from_yaml",
        classmethod(lambda cls, name="arena", section="sweep": SweepConfig(
            budget_timesteps=2048, control_style="direct", axes={"gamma": [0.95]},
            confirm_seeds=[0], confirm_top_k=1, summary_episodes=5,
            reward_hack_phase_tolerance=0.02)),
    )
    baseline = ArenaTrainConfig.from_yaml("arena")
    monkeypatch.setattr(
        ArenaTrainConfig, "from_yaml",
        classmethod(lambda cls, name="arena", section="training": type(baseline)(
            **{**baseline.__dict__, "n_envs": 1, "n_steps": 512, "batch_size": 64,
               "total_timesteps": 2048})),
    )

    models = tmp_path / "models"
    models.mkdir()
    incumbent = models / "ppo_direct.zip"
    incumbent.write_bytes(b"the model already shipped")

    sweep_arena.run_sweep(sweep_arena.parse_args([
        "--sweep-dir", str(tmp_path / "logs"),
        "--results-dir", str(tmp_path / "results"),
        "--models-dir", str(models),
    ]))

    assert incumbent.read_bytes() == b"the model already shipped", "incumbent was overwritten"
    assert (models / "ppo_direct_sweep.zip").exists(), "tuned model saved under its own name"
