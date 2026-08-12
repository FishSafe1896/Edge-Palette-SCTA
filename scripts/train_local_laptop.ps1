param(
  [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
  [ValidateSet("baseline", "ppeb_edge", "ppeb_freq", "ppeb_full", "ppeb_full_pcp", "ppeb_full_pcp_formal_10k", "ppeb_full_pcp_seed2026_20k")]
  [string]$RunName = "ppeb_full_pcp_seed2026_20k",
  [string]$ContentDir = "data\content\train",
  [string]$StyleDir = "data\ChinesePaperCutting",
  [string]$VggPath = "src\ChinesePaperCutting\ChinesePaperCutting_Transfer\pre_trained_models\vgg_normalised.pth",
  [int]$Epoch = 20000,
  [int]$BatchSize = 1,
  [double]$BaseLr = 0.0001,
  [double]$PcpWeight = 1.0,
  [double]$PcpEdgeWeight = 0.5,
  [double]$PcpFreqWeight = 0.3,
  [double]$PcpPaletteWeight = 0.5,
  [double]$PcpSmoothWeight = 0.0,
  [double]$PcpTextureWeight = 0.5,
  [int]$CheckpointSaveInterval = 2500,
  [int]$LatestCheckpointInterval = 400,
  [int]$LossCountInterval = 100,
  [int]$EarlyStopWarmup = 3000,
  [int]$EarlyStopPatience = 300,
  [double]$EarlyStopMinDelta = 0.1,
  [double]$EarlyStopSmoothing = 0.01,
  [int]$Seed = 2026,
  [string]$Python = ""
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

if (-not $Python) {
  $python39 = "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe"
  if (Test-Path $python39) {
    $Python = $python39
  } else {
    $Python = "python"
  }
}

foreach ($path in @($ContentDir, $StyleDir, $VggPath)) {
  if (-not (Test-Path $path)) {
    throw "Missing required path: $path"
  }
}

$argsByRun = @{
  "baseline" = @("--ppeb_mode", "off")
  "ppeb_edge" = @("--ppeb_mode", "edge")
  "ppeb_freq" = @("--ppeb_mode", "freq")
  "ppeb_full" = @("--ppeb_mode", "full")
  "ppeb_full_pcp" = @("--ppeb_mode", "full", "--use_pcp_loss", "--pcp_weight", "$PcpWeight")
  "ppeb_full_pcp_formal_10k" = @("--ppeb_mode", "full", "--use_pcp_loss", "--pcp_weight", "$PcpWeight")
  "ppeb_full_pcp_seed2026_20k" = @("--ppeb_mode", "full", "--use_pcp_loss", "--pcp_weight", "$PcpWeight")
}

$checkpointDir = "outputs\checkpoints\$RunName"
New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null

& $Python "src\ChinesePaperCutting\ChinesePaperCutting_Transfer\train.py" `
  --content_dir $ContentDir `
  --style_dir $StyleDir `
  --vgg_dir $VggPath `
  --epoch $Epoch `
  --batch_size $BatchSize `
  --seed $Seed `
  --base_lr $BaseLr `
  --pcp_edge_weight $PcpEdgeWeight `
  --pcp_freq_weight $PcpFreqWeight `
  --pcp_palette_weight $PcpPaletteWeight `
  --pcp_smooth_weight $PcpSmoothWeight `
  --pcp_texture_weight $PcpTextureWeight `
  --checkpoint_save_interval $CheckpointSaveInterval `
  --latest_checkpoint_interval $LatestCheckpointInterval `
  --loss_count_interval $LossCountInterval `
  --auto_resume `
  --early_stop_warmup $EarlyStopWarmup `
  --early_stop_patience $EarlyStopPatience `
  --early_stop_min_delta $EarlyStopMinDelta `
  --early_stop_smoothing $EarlyStopSmoothing `
  --checkpoint_save_path $checkpointDir `
  @($argsByRun[$RunName])

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
