import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from diffusion_policy.common.gaze_wam_training_config import (
    gaze_wam_prepared_dataloader_batches,
    gaze_wam_planned_optimizer_steps,
    resolve_gaze_wam_batching_config,
    validate_gaze_wam_training_config,
)
from diffusion_policy.common.gaze_wam_transfer import (
    export_gaze_wam_transfer_artifact,
    load_gaze_wam_transfer_artifact,
)
from diffusion_policy.dataset.gaze_wam_batching import build_gaze_wam_dataloader
from diffusion_policy.scripts.plan_gaze_wam_experiments import (
    _default_training_stage_for_config,
    _default_variants,
    _job_provenance,
    _sweep_variants,
)
from diffusion_policy.workspace.train_gaze_wam_workspace import (
    _validate_gaze_wam_accumulation_flush_contract,
)


class _TinyDataset(Dataset):
    def __init__(self, size):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"value": torch.tensor(index, dtype=torch.int64)}


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder = torch.nn.Linear(3, 4)
        self.gaze_encoder = torch.nn.Linear(4, 2)


def _cfg(robot_batch=48, open_batch=16, source="ratio"):
    return OmegaConf.create(
        {
            "robot_dataloader": {"batch_size": robot_batch},
            "open_dataloader": {"batch_size": open_batch},
            "data_mixing": {
                "batch_size_source": source,
                "total_batch_size_per_process": robot_batch + open_batch,
                "robot_ratio": robot_batch / (robot_batch + open_batch),
                "open_ratio": open_batch / (robot_batch + open_batch),
            },
        }
    )


def _stage_cfg(
    stage,
    robot_batch,
    open_batch,
    *,
    val_robot_batch=None,
    val_open_batch=0,
    max_train_steps=None,
    export_path="",
):
    total = robot_batch + open_batch
    return OmegaConf.create(
        {
            "robot_dataloader": {
                "batch_size": robot_batch,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "drop_last": False,
            },
            "open_dataloader": {
                "batch_size": open_batch,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "drop_last": False,
            },
            "val_robot_dataloader": {
                "batch_size": robot_batch if val_robot_batch is None else val_robot_batch,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "drop_last": False,
            },
            "val_open_dataloader": {
                "batch_size": val_open_batch,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "drop_last": False,
            },
            "data_mixing": {
                "batch_size_source": "ratio",
                "total_batch_size_per_process": total,
                "robot_ratio": robot_batch / total if total else 0.0,
                "open_ratio": open_batch / total if total else 0.0,
            },
            "training": {
                "stage": stage,
                "gradient_accumulate_every": 1,
                "num_epochs": 1,
                "checkpoint_every": 1,
                "val_every": 1,
                "sample_every": 1,
                "gdr_every": 0,
                "max_train_steps": max_train_steps,
                "max_val_steps": None,
                "lr_warmup_steps": 0,
                "tqdm_interval_sec": 0.0,
                "resume": False,
                "transfer": {
                    "load_path": "",
                    "load_scope": "obs_encoder",
                    "export_path": export_path,
                    "export_scope": "obs_encoder",
                    "export_overwrite": False,
                },
            },
            "checkpoint": {"topk": {"monitor_key": "val_robot_loss"}},
        }
    )


def test_ratio_batching_is_robot_driven_and_rounds_source_quota():
    summary = resolve_gaze_wam_batching_config(_cfg(5, 2))

    assert summary["valid"] is True
    assert summary["resolved_batch_size_source"] == "ratio"
    assert summary["robot_batch_size"] == 5
    assert summary["open_batch_size"] == 2
    assert summary["robot_ratio"] == pytest.approx(5 / 7)
    assert summary["open_ratio"] == pytest.approx(2 / 7)
    assert summary["robot_tail_policy"] == "pad"


def test_auto_source_preserves_legacy_dataloader_override():
    cfg = _cfg(48, 16, source="auto")
    cfg.robot_dataloader.batch_size = 6
    cfg.open_dataloader.batch_size = 2

    summary = resolve_gaze_wam_batching_config(cfg)

    assert summary["valid"] is True
    assert summary["compatibility_fallback_to_dataloader"] is True
    assert summary["resolved_batch_size_source"] == "dataloader"
    assert (summary["robot_batch_size"], summary["open_batch_size"]) == (6, 2)


def test_ratio_batching_rejects_inconsistent_ratios():
    cfg = _cfg(3, 1)
    cfg.data_mixing.open_ratio = 0.4

    summary = resolve_gaze_wam_batching_config(cfg)

    assert summary["valid"] is False
    assert any("must equal 1.0" in error for error in summary["errors"])


def test_stage_contract_requires_explicit_source_and_robot_validation_roles():
    mixed_without_open = validate_gaze_wam_training_config(
        _stage_cfg("mixed_train", 4, 0)
    )
    assert not mixed_without_open["valid"]
    assert any("positive open-source quota" in error for error in mixed_without_open["errors"])

    robot_without_val = validate_gaze_wam_training_config(
        _stage_cfg("robot_only", 4, 0, val_robot_batch=0)
    )
    assert not robot_without_val["valid"]
    assert any("val_robot_dataloader.batch_size" in error for error in robot_without_val["errors"])

    valid_robot = validate_gaze_wam_training_config(_stage_cfg("robot_only", 4, 0))
    assert valid_robot["valid"] is True


def test_open_pretrain_requires_fixed_budget_export_and_validation():
    invalid = validate_gaze_wam_training_config(
        _stage_cfg("open_pretrain", 0, 4, val_open_batch=0)
    )
    assert not invalid["valid"]
    assert any("max_train_steps" in error for error in invalid["errors"])
    assert any("export_path" in error for error in invalid["errors"])
    assert any("val_open_dataloader.batch_size" in error for error in invalid["errors"])

    valid = validate_gaze_wam_training_config(
        _stage_cfg(
            "open_pretrain",
            0,
            4,
            val_open_batch=4,
            max_train_steps=20,
            export_path="data/pretrain.pt",
        )
    )
    assert valid["valid"] is True


def test_planned_optimizer_steps_flushes_accumulation_per_epoch():
    assert gaze_wam_planned_optimizer_steps(
        steps_per_epoch=5,
        num_epochs=2,
        gradient_accumulate_every=2,
    ) == 6
    assert gaze_wam_planned_optimizer_steps(
        steps_per_epoch=5,
        num_epochs=2,
        gradient_accumulate_every=2,
        max_train_steps=4,
    ) == 4


def test_prepared_batch_count_matches_accelerate_sharding_modes():
    assert gaze_wam_prepared_dataloader_batches(
        5, num_processes=2, split_batches=False, even_batches=True, drop_last=False
    ) == 3
    assert gaze_wam_prepared_dataloader_batches(
        5, num_processes=2, split_batches=False, even_batches=True, drop_last=True
    ) == 2
    assert gaze_wam_prepared_dataloader_batches(
        5, num_processes=2, split_batches=False, even_batches=False, process_index=0
    ) == 3
    assert gaze_wam_prepared_dataloader_batches(
        5, num_processes=2, split_batches=False, even_batches=False, process_index=1
    ) == 2
    assert gaze_wam_prepared_dataloader_batches(
        5, num_processes=2, split_batches=True, even_batches=True, drop_last=True
    ) == 5


def test_accumulation_contract_requires_epoch_boundary_flush():
    enabled = SimpleNamespace(
        gradient_state=SimpleNamespace(sync_with_dataloader=True)
    )
    assert _validate_gaze_wam_accumulation_flush_contract(enabled) is True

    disabled = SimpleNamespace(
        gradient_state=SimpleNamespace(sync_with_dataloader=False)
    )
    with pytest.raises(RuntimeError, match="sync_with_dataloader=true"):
        _validate_gaze_wam_accumulation_flush_contract(disabled)


def test_accelerate_flushes_partial_accumulation_at_epoch_end():
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, TensorDataset

    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=3)
    model = torch.nn.Linear(1, 1)
    dataloader = DataLoader(
        TensorDataset(torch.arange(5, dtype=torch.float32).reshape(5, 1)),
        batch_size=1,
    )
    dataloader, model = accelerator.prepare(dataloader, model)
    try:
        sync_flags = []
        for _batch in dataloader:
            with accelerator.accumulate(model):
                sync_flags.append(bool(accelerator.sync_gradients))
        assert sync_flags == [False, False, True, False, True]
    finally:
        accelerator.free_memory(dataloader, model)


def test_provenance_uses_ratio_quota_as_authority():
    row = _job_provenance(
        "train_gaze_wam_workspace",
        plan_debug=False,
        overrides=[
            "robot_dataloader.batch_size=8",
            "open_dataloader.batch_size=2",
        ],
    )
    assert row["batch_size_source"] == "ratio"
    assert (row["robot_batch_size"], row["open_batch_size"]) == (48, 16)
    assert row["total_batch_size_per_process"] == 64
    assert row["robot_ratio"] == pytest.approx(0.75)
    assert row["open_ratio"] == pytest.approx(0.25)


def test_debug_auto_provenance_keeps_explicit_dataloader_override():
    row = _job_provenance(
        "train_gaze_wam_debug_workspace",
        plan_debug=True,
        overrides=[
            "robot_dataloader.batch_size=6",
            "open_dataloader.batch_size=2",
        ],
    )
    assert row["batch_size_source"] == "dataloader"
    assert row["requested_batch_size_source"] == "auto"
    assert (row["robot_batch_size"], row["open_batch_size"]) == (6, 2)
    assert row["total_batch_size_per_process"] == 8


def test_default_experiment_plan_has_robot_baseline_before_mixed_training():
    full = _default_variants(debug=False)
    debug = _default_variants(debug=True)

    assert full[0]["config"] == "train_gaze_wam_robot_only_workspace"
    assert full[1]["config"] == "train_gaze_wam_workspace"
    assert [row["config"] for row in full] == [
        "train_gaze_wam_robot_only_workspace",
        "train_gaze_wam_workspace",
    ]
    assert debug[0]["config"] == "train_gaze_wam_robot_only_debug_workspace"
    assert debug[1]["config"] == "train_gaze_wam_debug_workspace"
    assert not any("open_only" in row["config"] for row in full + debug)


def test_ratio_sweep_uses_robot_only_config_for_zero_open_quota():
    rows = _sweep_variants("open_ratio", debug=False)
    assert rows[0]["config"] == "train_gaze_wam_robot_only_workspace"
    assert rows[0]["overrides"][-4:] == [
        "robot_dataloader.batch_size=64",
        "open_dataloader.batch_size=0",
        "val_robot_dataloader.batch_size=64",
        "val_open_dataloader.batch_size=0",
    ]
    assert rows[1]["config"] == "train_gaze_wam_workspace"


def test_planner_stage_inference_keeps_optional_pretrain_explicit():
    assert (
        _default_training_stage_for_config(
            "train_gaze_wam_open_pretrain_workspace"
        )
        == "open_pretrain"
    )
    assert (
        _default_training_stage_for_config("train_gaze_wam_robot_only_workspace")
        == "robot_only"
    )


def test_tail_policies_keep_drop_and_pad_source_batch_sizes():
    cfg = {
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
        "persistent_workers": False,
    }

    keep = build_gaze_wam_dataloader(
        _TinyDataset(5), cfg, batch_size=3, tail_policy="keep", source_name="keep"
    )
    drop = build_gaze_wam_dataloader(
        _TinyDataset(5), cfg, batch_size=3, tail_policy="drop", source_name="drop"
    )
    pad = build_gaze_wam_dataloader(
        _TinyDataset(5), cfg, batch_size=3, tail_policy="pad", source_name="pad"
    )

    assert [int(batch["value"].shape[0]) for batch in keep] == [3, 2]
    assert [int(batch["value"].shape[0]) for batch in drop] == [3]
    assert [int(batch["value"].shape[0]) for batch in pad] == [3, 3]
    assert pad.batch_size == 3


def test_transfer_artifact_round_trip_and_scope_guard(tmp_path):
    source = _TinyPolicy()
    target = _TinyPolicy()
    original = copy.deepcopy(target.obs_encoder.state_dict())
    path = tmp_path / "obs_encoder.pt"

    exported = export_gaze_wam_transfer_artifact(
        source,
        path,
        scope="obs_encoder",
        metadata={"stage": "open_pretrain"},
    )
    loaded = load_gaze_wam_transfer_artifact(target, path, scope="obs_encoder")

    assert exported["scope"] == "obs_encoder"
    assert loaded["metadata"]["stage"] == "open_pretrain"
    assert all(
        torch.equal(value, source.obs_encoder.state_dict()[key])
        for key, value in target.obs_encoder.state_dict().items()
    )
    assert any(
        not torch.equal(value, source.obs_encoder.state_dict()[key])
        for key, value in original.items()
    )

    with pytest.raises(ValueError, match="does not match"):
        load_gaze_wam_transfer_artifact(target, path, scope="obs_and_gaze")
