# Gaze-WAM Full-Resolution Heatmap Review

Date: 2026-06-07

## Scope

This review records the correction after the dim4/dim16 compact-latent runs, the later
`heatmap_dim=256` lossless patch run, and the interim learned-decoder run were stopped or demoted to
historical comparisons. The target path is still full-resolution heatmap supervision from
`gaze_xy`, but the active model-side representation is now frozen-Cosmos latent diffusion plus a
frozen Cosmos decoder.

## FastWAM Alignment

FastWAM denoises video in VAE latent space, not raw full-resolution RGB pixels. RGB/video exists at
the data boundary and after VAE decode; the DiT operates on latent video tensors/tokens.

Gaze-WAM mirrors that contract as closely as practical for a 1-channel gaze heatmap:

- The heatmap DiT denoises compact heatmap latent tokens.
- The frozen NVIDIA Cosmos `Cosmos-Tokenizer-CI16x16` decoder maps denormalized predicted latents to
  full-resolution `256 x 256` logits.
- DSNT coordinate loss, spatial JS regularization, previews, and paper figures use the decoded
  full-resolution heatmap.

## Current Heatmap Contract

- Dense target: online `256 x 256 x 1` Gaussian distribution from normalized `gaze_xy`.
- Token grid: `16 x 16 = 256`.
- Latent channels per token: `heatmap_dim=16`.
- Latent image: frozen Cosmos continuous latent `[C=16, H=16, W=16]`.
- Codec: frozen Cosmos CI16x16 encoder creates clean latent labels; frozen Cosmos CI16x16 decoder
  maps predicted clean latents back to `256 x 256` logits. Decoder weights stay frozen, but decoder
  inputs are not wrapped in `no_grad`, so DSNT/JS gradients flow back to the heatmap DiT.
- Latent scale: the standalone `Cosmos-0.1-Tokenizer-CI16x16` JIT package does not provide a usable
  checkpoint-specific normalizer. Gaze-WAM uses the project-estimated scale
  `heatmap_latent_scale=0.25`, `heatmap_latent_offset=0.0`, from
  `data/outputs/cosmos_heatmap_latent_stats/hot3d_open_ci16x16_random4096_seed42.json`. This maps
  the observed raw label range `[-3.921875, 3.375]` to `[-0.98046875, 0.84375]`, so scheduler
  `clip_sample=[-1, 1]` constrains runaway predictions without clipping clean labels.
- Zarr contract: `data/gaze_xy` is required; `data/gaze_heatmap` is optional metadata and not
  required for the main DSNT/JS path.

The stopped `heatmap_dim=256` patch path is now considered an obsolete transitional run: it could
recover full-resolution pixels exactly, but it directly packed image patches into tokens instead of
using a pretrained tokenizer latent space.

## Training Contract

- Open rows train heatmap only under the learned `[MASK]` gaze token.
- Robot real-gaze rows train action only.
- Robot gaze-dropout rows may train action plus heatmap.
- Action inference omits heatmap target tokens and reuses the stable image/gaze world K/V cache.
- AMP is mandatory for non-debug training; server launch should use 8-GPU bf16 Accelerate.

## Remaining Risk

Cosmos CI16x16 is pretrained and frozen, but it was trained as an RGB image tokenizer rather than a
specialized 1-channel heatmap/mask codec. Future ablations should compare CI8x8, other SOTA image
codecs, or a heatmap/mask-specific frozen codec if the decoded heatmap still shows texture artifacts.
