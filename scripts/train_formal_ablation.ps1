param(
  [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
  [string]$ContentDir = "data\content\train",
  [string]$StyleDir = "data\ChinesePaperCutting",
  [string]$VggPath = "src\ChinesePaperCutting\ChinesePaperCutting_Transfer\pre_trained_models\vgg_normalised.pth",
  [int]$Epoch = 40000,
  [int]$BatchSize = 1,
  [double]$BaseLr = 0.0001,
  [int]$Seed = 2026,
  [switch]$Smoke,
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

$train = "src\ChinesePaperCutting\ChinesePaperCutting_Transfer\train.py"
$prefix = if ($Smoke) { "_smoke_" } else { "" }
$effectiveEpoch = if ($Smoke) { 1 } else { $Epoch }
$checkpointInterval = if ($Smoke) { 1000000 } else { 40000 }
$latestInterval = if ($Smoke) { 0 } else { 400 }
$earlyStopWarmup = if ($Smoke) { 0 } else { 8000 }
$earlyStopPatience = if ($Smoke) { 0 } else { 1000 }

$sharedArgs = @(
  "--content_dir", $ContentDir,
  "--style_dir", $StyleDir,
  "--vgg_dir", $VggPath,
  "--epoch", "$effectiveEpoch",
  "--batch_size", "$BatchSize",
  "--seed", "$Seed",
  "--base_lr", "$BaseLr",
  "--checkpoint_save_interval", "$checkpointInterval",
  "--latest_checkpoint_interval", "$latestInterval",
  "--loss_count_interval", "400",
  "--best_checkpoint_metric", "step",
  "--auto_resume",
  "--early_stop_warmup", "$earlyStopWarmup",
  "--early_stop_patience", "$earlyStopPatience",
  "--early_stop_min_delta", "0.05",
  "--early_stop_smoothing", "0.01",
  "--content_weight", "2.5",
  "--style_weight", "2.0",
  "--id1_weight", "80",
  "--id2_weight", "1"
)

$pcpFullArgs = @(
  "--use_pcp_loss",
  "--pcp_weight", "1.0",
  "--pcp_warmup_steps", "4000",
  "--pcp_edge_weight", "1.0",
  "--pcp_freq_weight", "10",
  "--pcp_palette_weight", "30",
  "--pcp_smooth_weight", "0.0",
  "--pcp_texture_weight", "20"
)

$runs = @(
  @{
    Name = "${prefix}ablation_wo_edge_prior_seed2026_40k"
    Args = @(
      "--ppeb_mode", "freq",
      "--use_style_adapter",
      "--style_adapter_alpha", "0.1",
      "--use_pcp_loss",
      "--pcp_weight", "1.0",
      "--pcp_warmup_steps", "4000",
      "--pcp_edge_weight", "0.0",
      "--pcp_freq_weight", "10",
      "--pcp_palette_weight", "30",
      "--pcp_smooth_weight", "0.0",
      "--pcp_texture_weight", "20"
    )
  },
  @{
    Name = "${prefix}ablation_wo_frequency_prior_seed2026_40k"
    Args = @(
      "--ppeb_mode", "edge",
      "--use_style_adapter",
      "--style_adapter_alpha", "0.1",
      "--use_pcp_loss",
      "--pcp_weight", "1.0",
      "--pcp_warmup_steps", "4000",
      "--pcp_edge_weight", "1.0",
      "--pcp_freq_weight", "0.0",
      "--pcp_palette_weight", "30",
      "--pcp_smooth_weight", "0.0",
      "--pcp_texture_weight", "0.0"
    )
  },
  @{
    Name = "${prefix}ablation_wo_scta_seed2026_40k"
    Args = @("--ppeb_mode", "full") + $pcpFullArgs
  },
  @{
    Name = "${prefix}ablation_wo_cut_loss_seed2026_40k"
    Args = @(
      "--ppeb_mode", "full",
      "--use_style_adapter",
      "--style_adapter_alpha", "0.1"
    )
  }
)

foreach ($run in $runs) {
  $checkpointDir = "outputs\checkpoints\$($run.Name)"
  $logPrefix = "outputs\logs\train_$($run.Name)"
  New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null
  New-Item -ItemType Directory -Force -Path "outputs\logs" | Out-Null

  $args = @("-u", $train) + $sharedArgs + @("--checkpoint_save_path", $checkpointDir) + $run.Args
  $command = "$Python " + (($args | ForEach-Object {
    if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
  }) -join " ")
  Set-Content -Path "$logPrefix.command.txt" -Value $command -Encoding ASCII

  Write-Host "Starting $($run.Name)"
  $proc = Start-Process -FilePath $Python `
    -ArgumentList $args `
    -RedirectStandardOutput "$logPrefix`_stdout.log" `
    -RedirectStandardError "$logPrefix`_stderr.log" `
    -NoNewWindow `
    -PassThru
  Set-Content -Path "$logPrefix.pid" -Value $proc.Id -Encoding ASCII
  $proc.WaitForExit()
  $proc.Refresh()
  $exitCode = $proc.ExitCode
  if ($null -eq $exitCode) {
    $exitCode = 0
  }
  if ($exitCode -ne 0) {
    throw "Training run failed: $($run.Name), exit code $exitCode. See $logPrefix`_stderr.log"
  }
}
