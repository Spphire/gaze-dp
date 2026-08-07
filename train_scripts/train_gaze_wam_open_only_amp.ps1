param(
    [string]$OutputDir = "data/outputs/hot3d_open_cosmos_intensity_softplus_latent_amp",
    [int]$MaxTrainSteps = 0,
    [int]$MaxValSteps = 20,
    [int]$NumEpochs = 200,
    [int]$BatchSize = 16,
    [int]$GradientAccumulateEvery = 4,
    [int]$HeatmapDim = 16,
    [string]$HeatmapObjective = "dsnt_js",
    [bool]$Pretrained = $false,
    [string]$LoggingMode = "disabled",
    [string]$CosmosEncoderPath = "data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/encoder.jit",
    [string]$CosmosDecoderPath = "data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/decoder.jit",
    [double]$HeatmapLatentScale = 0.25,
    [double]$HeatmapLatentOffset = 0.0,
    [string]$HeatmapLatentStatsPath = "data/outputs/cosmos_heatmap_latent_stats/hot3d_open_ci16x16_random4096_seed42.json",
    [bool]$HeatmapSchedulerClipSample = $true,
    [string]$HeatmapDistributionMode = "intensity_softplus",
    [double]$HeatmapDsntTemperature = 0.1
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Accelerate = Join-Path $Root ".venv\Scripts\accelerate.exe"
if (-not (Test-Path $Accelerate)) {
    throw "Missing accelerate executable at $Accelerate. Create the project venv first."
}

if (-not (Test-Path $CosmosEncoderPath)) {
    throw "Missing Cosmos encoder JIT at $CosmosEncoderPath. Run diffusion_policy/scripts/download_cosmos_tokenizer.py first."
}

if (-not (Test-Path $CosmosDecoderPath)) {
    throw "Missing Cosmos decoder JIT at $CosmosDecoderPath. Run diffusion_policy/scripts/download_cosmos_tokenizer.py first."
}

$Args = @(
    "launch",
    "--num_processes", "1",
    "--mixed_precision", "bf16",
    "train.py",
    "--config-name=train_gaze_wam_open_only_cosmos_workspace",
    "task.heatmap_dim=$HeatmapDim",
    "policy.heatmap_dim=$HeatmapDim",
    "policy.heatmap_spatial_decoder=cosmos_tokenizer",
    "policy.heatmap_objective=$HeatmapObjective",
    "policy.heatmap_cosmos_encoder_path=$CosmosEncoderPath",
    "policy.heatmap_cosmos_decoder_path=$CosmosDecoderPath",
    "policy.heatmap_latent_scale=$HeatmapLatentScale",
    "policy.heatmap_latent_offset=$HeatmapLatentOffset",
    "policy.heatmap_latent_stats_path=$HeatmapLatentStatsPath",
    "policy.heatmap_scheduler_clip_sample=$($HeatmapSchedulerClipSample.ToString().ToLowerInvariant())",
    "policy.heatmap_distribution_mode=$HeatmapDistributionMode",
    "policy.heatmap_dsnt_temperature=$HeatmapDsntTemperature",
    "policy.obs_encoder.pretrained=$($Pretrained.ToString().ToLowerInvariant())",
    "data_mixing.batch_size_source=ratio",
    "data_mixing.total_batch_size_per_process=$BatchSize",
    "data_mixing.robot_ratio=0.0",
    "data_mixing.open_ratio=1.0",
    "open_dataloader.batch_size=$BatchSize",
    "val_open_dataloader.batch_size=$BatchSize",
    "training.gradient_accumulate_every=$GradientAccumulateEvery",
    "training.num_epochs=$NumEpochs",
    "training.max_val_steps=$MaxValSteps",
    "logging.mode=$LoggingMode",
    "hydra.run.dir=$OutputDir"
)

if ($MaxTrainSteps -gt 0) {
    $Args += "training.max_train_steps=$MaxTrainSteps"
}

& $Accelerate @Args
