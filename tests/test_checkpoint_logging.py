import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "ChinesePaperCutting" / "ChinesePaperCutting_Transfer"
sys.path.insert(0, str(MODEL_DIR))

from tools import save_checkpoint
from tools import LossImprovementTracker


def test_save_checkpoint_persists_extra_loss_logs(tmp_path):
    path = tmp_path / "checkpoint.pkl"

    save_checkpoint(
        encoder=None,
        transModule=None,
        decoder=None,
        optimizer=None,
        scheduler=None,
        epoch=4,
        log_c=[1.0],
        log_s=[2.0],
        log_id1=[3.0],
        log_id2=[4.0],
        log_all=[5.0],
        loss_count_interval=1,
        save_path=path,
        extra_logs={"log_pcp_total": [0.25], "log_pcp_edge": [0.1]},
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["log_all"] == [5.0]
    assert checkpoint["log_pcp_total"] == [0.25]
    assert checkpoint["log_pcp_edge"] == [0.1]


def test_save_checkpoint_persists_best_metric_metadata(tmp_path):
    path = tmp_path / "best_checkpoint.pkl"

    save_checkpoint(
        encoder=None,
        transModule=None,
        decoder=None,
        optimizer=None,
        scheduler=None,
        epoch=18800,
        log_c=[],
        log_s=[],
        log_id1=[],
        log_id2=[],
        log_all=[21.54],
        loss_count_interval=100,
        save_path=path,
        extra_state={
            "best_metric_name": "loss_all",
            "best_metric_value": 21.54,
            "best_metric_step": 18800,
        },
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["best_metric_name"] == "loss_all"
    assert checkpoint["best_metric_value"] == 21.54
    assert checkpoint["best_metric_step"] == 18800


def test_save_checkpoint_keeps_previous_file_when_atomic_write_fails(tmp_path, monkeypatch):
    path = tmp_path / "best_checkpoint.pkl"
    path.write_bytes(b"previous-good-checkpoint")

    def fail_save(checkpoint, save_path):
        Path(save_path).write_bytes(b"partial")
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(torch, "save", fail_save)

    try:
        save_checkpoint(
            encoder=None,
            transModule=None,
            decoder=None,
            optimizer=None,
            scheduler=None,
            epoch=1,
            log_c=[],
            log_s=[],
            log_id1=[],
            log_id2=[],
            log_all=[],
            loss_count_interval=1,
            save_path=path,
        )
    except RuntimeError:
        pass

    assert path.read_bytes() == b"previous-good-checkpoint"


def test_loss_tracker_saves_best_on_any_loss_decrease():
    tracker = LossImprovementTracker(best_loss=10.0, early_stop_min_delta=0.2)

    result = tracker.update(current_loss=9.99, step=1)

    assert result.is_best is True
    assert result.best_loss == 9.99
    assert result.best_step == 1


def test_loss_tracker_early_stop_uses_min_delta_and_patience_per_step():
    tracker = LossImprovementTracker(
        best_loss=10.0,
        early_stop_best_loss=10.0,
        early_stop_min_delta=0.2,
        early_stop_patience=2,
    )

    first = tracker.update(current_loss=9.95, step=1)
    second = tracker.update(current_loss=9.94, step=2)

    assert first.is_best is True
    assert first.should_stop is False
    assert first.bad_steps == 1
    assert second.is_best is True
    assert second.should_stop is True
    assert second.bad_steps == 2


def test_loss_tracker_does_not_early_stop_before_warmup():
    tracker = LossImprovementTracker(
        best_loss=10.0,
        early_stop_best_loss=10.0,
        early_stop_min_delta=0.2,
        early_stop_patience=1,
        early_stop_warmup=5,
    )

    result = tracker.update(current_loss=10.5, step=4)

    assert result.should_stop is False
    assert result.bad_steps == 0


def test_loss_tracker_can_use_smoothed_loss_for_early_stop_without_changing_best():
    tracker = LossImprovementTracker(
        best_loss=100.0,
        early_stop_best_loss=100.0,
        early_stop_min_delta=0.1,
        early_stop_patience=2,
        early_stop_smoothing=0.1,
        early_stop_monitor_loss=100.0,
    )

    outlier = tracker.update(current_loss=20.0, step=1)
    rebound = tracker.update(current_loss=70.0, step=2)
    lower_than_rebound = tracker.update(current_loss=65.0, step=3)

    assert outlier.is_best is True
    assert outlier.best_loss == 20.0
    assert outlier.early_stop_best_loss == 92.0
    assert rebound.bad_steps == 0
    assert rebound.early_stop_best_loss == 89.8
    assert lower_than_rebound.should_stop is False
    assert lower_than_rebound.early_stop_best_loss < outlier.early_stop_best_loss


def test_loss_tracker_does_not_seed_smoothed_early_stop_from_raw_best_without_monitor():
    tracker = LossImprovementTracker(
        best_loss=20.0,
        early_stop_min_delta=0.1,
        early_stop_patience=2,
        early_stop_smoothing=0.1,
    )

    result = tracker.update(current_loss=60.0, step=1)

    assert result.best_loss == 20.0
    assert result.early_stop_monitor_loss == 60.0
    assert result.early_stop_best_loss == 60.0
    assert result.bad_steps == 0


def test_train_defaults_best_checkpoint_to_logged_average_loss():
    train_source = (MODEL_DIR / "train.py").read_text(encoding="utf-8")

    assert "--best_checkpoint_metric" in train_source
    assert "default='log_avg'" in train_source
    assert "loss_all_log_avg" in train_source
