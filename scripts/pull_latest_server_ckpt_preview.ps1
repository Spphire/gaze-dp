param(
    [string]$Server = "root@106.14.2.243",
    [int]$Port = 1024,
    [string]$RemoteProject = "/mnt/workspace/shenyibo/gaze-wam",
    [string]$RemoteOutput = "data/outputs/hot3d_open_latent_cached_8gpu_amp_20260606_164432",
    [string]$LocalDir = ".codex_tmp/server_watched_preview_latest"
)

$ErrorActionPreference = "Stop"

$remotePreviewRoot = "$RemoteProject/$RemoteOutput/media/ckpt_heatmap/watched"
$latestRemoteLines = @(& ssh -p $Port $Server "find '$remotePreviewRoot' -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-")
$latestRemoteDir = ($latestRemoteLines | Select-Object -Last 1)
if ($null -ne $latestRemoteDir) {
    $latestRemoteDir = $latestRemoteDir.Trim()
}

if ([string]::IsNullOrWhiteSpace($latestRemoteDir)) {
    Write-Host "No watched checkpoint preview directory exists yet."
    Write-Host "Remote preview root: $remotePreviewRoot"
    exit 1
}

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

$files = @("comparison.png", "summary.json", "preview.log")
foreach ($file in $files) {
    $remoteExists = & ssh -p $Port $Server "test -f '$latestRemoteDir/$file'"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Skipping missing remote file: $latestRemoteDir/$file"
        continue
    }
    $remoteFile = "${Server}:$latestRemoteDir/$file"
    $localFile = Join-Path $LocalDir $file
    & scp -P $Port $remoteFile $localFile
}

Write-Host "Pulled latest checkpoint preview:"
Write-Host "  remote: $latestRemoteDir"
Write-Host "  local:  $((Resolve-Path $LocalDir).Path)"
