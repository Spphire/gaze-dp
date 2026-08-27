import torch

from diffusion_policy.model.gaze_wam.cached_dual_stream_transformer import (
    CachedDualStreamGazeWamTransformer,
    CachedMixedAttention,
)


def test_target_attention_uses_independent_single_token_gaze_branch() -> None:
    torch.manual_seed(7)
    attention = CachedMixedAttention(n_emb=8, n_head=2, p_drop_attn=0.5)
    attention.eval()
    target = torch.randn(2, 3, 8)
    image_key = torch.randn(2, 2, 5, 4)
    image_value = torch.randn(2, 2, 5, 4)
    gaze_key = torch.randn(2, 2, 1, 4)
    gaze_value = torch.randn(2, 2, 1, 4)

    # The branch has exactly one source token, so its pre-dropout softmax is 1.
    query = attention._reshape_heads(attention.query(target))
    gaze_logits = torch.matmul(query, gaze_key.transpose(-2, -1)) * attention.scale
    gaze_weights = torch.softmax(gaze_logits, dim=-1)
    assert gaze_weights.shape == (2, 2, 3, 1)
    torch.testing.assert_close(gaze_weights, torch.ones_like(gaze_weights))

    output = attention(target, image_key, image_value, gaze_key, gaze_value)
    assert torch.isfinite(output).all()


def test_gaze_kv_ablation_changes_independent_attention_output() -> None:
    torch.manual_seed(11)
    attention = CachedMixedAttention(n_emb=8, n_head=2, p_drop_attn=0.0)
    attention.eval()
    target = torch.randn(2, 3, 8)
    image_key = torch.randn(2, 2, 5, 4)
    image_value = torch.randn(2, 2, 5, 4)
    gaze_key = torch.randn(2, 2, 1, 4)
    gaze_value = torch.randn(2, 2, 1, 4)

    with_gaze = attention(target, image_key, image_value, gaze_key, gaze_value)
    without_gaze = attention(
        target,
        image_key,
        image_value,
        gaze_key,
        torch.zeros_like(gaze_value),
    )
    assert torch.isfinite(with_gaze).all()
    assert torch.linalg.vector_norm(with_gaze - without_gaze).item() > 1e-6


def test_world_cache_contract_has_separate_image_and_gaze_sources() -> None:
    model = CachedDualStreamGazeWamTransformer(
        action_dim=4,
        heatmap_dim=3,
        action_horizon=2,
        heatmap_num_tokens=2,
        max_image_tokens=5,
        n_layer=1,
        n_head=2,
        n_emb=8,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    model.eval()
    image_tokens = torch.randn(2, 5, 8)
    gaze_token = torch.randn(2, 1, 8)
    cache = model.prefill_world_cache(image_tokens, gaze_token)
    assert cache.image_key_values[0][0].shape[2] == 5
    assert cache.gaze_key_values[0][0].shape[2] == 1
    contract = model.attention_contract_summary(num_image_tokens=5)
    assert contract["target_attention_concatenates_image_gaze"] is False
    assert contract["target_attention_mode"] == "independent_image_gaze_target_softmax"
