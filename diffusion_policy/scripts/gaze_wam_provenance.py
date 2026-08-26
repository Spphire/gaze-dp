import hashlib
import json
from typing import Dict, Mapping, Sequence


PROVENANCE_CONTRACT_VERSION = "gaze_wam_policy_contract_v1"

PROVENANCE_CONTRACT_FIELDS = [
    "robot_batch_size",
    "open_batch_size",
    "robot_ratio",
    "open_ratio",
    "training_stage",
    "batch_size_source",
    "total_batch_size_per_process",
    "requested_robot_ratio",
    "requested_open_ratio",
    "gradient_accumulate_every",
    "num_processes",
    "mixed_precision",
    "distributed_type",
    "effective_robot_batch_size_per_optimizer_step",
    "effective_open_batch_size_per_optimizer_step",
    "effective_train_batch_size_per_optimizer_step",
    "robot_gaze_dropout_prob",
    "robot_heatmap_on_gaze_dropout",
    "robot_heatmap_supervision",
    "cfg_scale",
    "use_block_attention_mask",
    "heatmap_objective",
    "n_obs_steps",
    "action_horizon",
    "action_dim",
    "heatmap_num_tokens",
    "heatmap_token_grid",
    "image_shape",
    "image_resize_mode",
    "robot_image_resize_mode",
    "open_image_resize_mode",
    "obs_encoder_model_name",
    "obs_encoder_pretrained",
    "obs_encoder_checkpoint_path",
    "obs_encoder_cache_dir",
    "obs_encoder_local_weight_source_configured",
    "obs_encoder_local_weight_source_valid",
]


def provenance_contract_payload(
    row: Mapping[str, object],
    fields: Sequence[str] = PROVENANCE_CONTRACT_FIELDS,
) -> Dict[str, object]:
    """Return the normalized policy-training contract payload used for provenance ids."""
    return {
        "version": PROVENANCE_CONTRACT_VERSION,
        "fields": {field: row.get(field, "") for field in fields},
    }


def provenance_contract_id(
    row: Mapping[str, object],
    fields: Sequence[str] = PROVENANCE_CONTRACT_FIELDS,
    length: int = 16,
) -> str:
    """Create a stable short id for comparing train-plan and metric provenance rows."""
    if length <= 0:
        raise ValueError("length must be positive.")
    payload = provenance_contract_payload(row=row, fields=fields)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def add_provenance_contract(row: Mapping[str, object]) -> Dict[str, object]:
    """Copy a provenance row and add the contract version/id fields."""
    enriched = dict(row)
    enriched["provenance_contract_version"] = PROVENANCE_CONTRACT_VERSION
    enriched["provenance_contract_id"] = provenance_contract_id(enriched)
    return enriched
