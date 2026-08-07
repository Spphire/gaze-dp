from typing import Dict, Optional

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from diffusion_policy.common.gaze_wam_training_config import (
    normalize_gaze_wam_positive_int_field,
    normalize_gaze_wam_tail_policy,
)

class FixedSizeBatchCollator:
    """Pad only the final short source batch to keep mixed quotas stable."""

    def __init__(self, batch_size: int, source_name: str = "source") -> None:
        self.batch_size = normalize_gaze_wam_positive_int_field(
            f"{source_name}.batch_size",
            batch_size,
        )
        self.source_name = str(source_name)

    def __call__(self, samples):
        samples = list(samples)
        sample_count = len(samples)
        if sample_count <= 0:
            raise ValueError(f"{self.source_name} batch cannot be empty.")
        if sample_count > self.batch_size:
            raise ValueError(
                f"{self.source_name} batch has {sample_count} samples, exceeding "
                f"the configured quota {self.batch_size}."
            )
        if sample_count < self.batch_size:
            original = tuple(samples)
            missing = self.batch_size - sample_count
            samples.extend(original[idx % sample_count] for idx in range(missing))
        return default_collate(samples)


def gaze_wam_dataloader_kwargs(
    dataloader_cfg,
    *,
    batch_size: int,
    tail_policy: str,
    source_name: str,
    runtime_overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    batch_size = normalize_gaze_wam_positive_int_field(
        f"{source_name}.batch_size",
        batch_size,
    )
    tail_policy = normalize_gaze_wam_tail_policy(
        f"{source_name}.tail_policy",
        tail_policy,
    )
    if OmegaConf.is_config(dataloader_cfg):
        kwargs = OmegaConf.to_container(dataloader_cfg, resolve=True)
    else:
        kwargs = dict(dataloader_cfg)
    kwargs = dict(kwargs)
    kwargs["batch_size"] = batch_size
    kwargs["drop_last"] = tail_policy == "drop"
    kwargs.pop("collate_fn", None)
    if tail_policy == "pad":
        kwargs["collate_fn"] = FixedSizeBatchCollator(
            batch_size=batch_size,
            source_name=source_name,
        )
    if runtime_overrides:
        kwargs.update(runtime_overrides)
    return kwargs


def build_gaze_wam_dataloader(
    dataset,
    dataloader_cfg,
    *,
    batch_size: int,
    tail_policy: str,
    source_name: str,
    runtime_overrides: Optional[Dict[str, object]] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        **gaze_wam_dataloader_kwargs(
            dataloader_cfg,
            batch_size=batch_size,
            tail_policy=tail_policy,
            source_name=source_name,
            runtime_overrides=runtime_overrides,
        ),
    )
