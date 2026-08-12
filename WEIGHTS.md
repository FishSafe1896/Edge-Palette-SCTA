# Model Weights

The large parameter files are distributed in the [v1.0.0 GitHub Release](https://github.com/FishSafe1896/Edge-Palette-SCTA/releases/tag/v1.0.0). They are intentionally excluded from Git history. The release contains the final model and comparison-model weights only; ablation checkpoints are not included.

## Release Assets

| Asset | Contents |
| --- | --- |
| `EPS-SCTA_ours_best_weights.zip` | Final Edge-Palette-SCTA best checkpoint and VGG encoder weights |
| `StyTR-2_weights.zip` | StyTR-2 transformer, decoder, embedding, and VGG weights |
| `CAST_weights.zip` | CAST autoencoder, decoders, style VGG, and VGG weights |
| `S2WAT_weights.zip` | S2WAT checkpoint |
| `Aflutter-Craft_weights.zip` | Aflutter Craft VGG, transformer, and decoder weights |

## Extraction Map

The archives contain the original parameter filenames. Extract each archive into a temporary folder, then move the files to the target locations below.

```text
EPS-SCTA_ours_best_weights.zip
  best_checkpoint.pkl  -> checkpoints/ours_EPS-SCTA_best_checkpoint.pkl
  vgg_normalised.pth   -> weights/vgg_normalised.pth

StyTR-2_weights.zip
  transformer_iter_160000.pth
  decoder_iter_160000.pth
  embedding_iter_160000.pth
  vgg_normalised.pth
  -> src/baselines/StyTR-2/experiments/ (rename the VGG file only if needed)

CAST_weights.zip
  latest_net_AE.pth
  latest_net_Dec_A.pth
  latest_net_Dec_B.pth
  style_vgg.pth
  vgg_normalised.pth
  -> src/baselines/CAST_pytorch/checkpoints/CAST_model/ for the three network files;
     put style_vgg.pth and vgg_normalised.pth in src/baselines/CAST_pytorch/models/

S2WAT_weights.zip
  checkpoint_40000_epoch.pkl -> src/baselines/S2WAT/pre_trained_models/checkpoint/

Aflutter-Craft_weights.zip
  vgg.pth, transformer.pth, decoder.pth
  -> external/Aflutter-Craft-API/models/
```

For the final model, the supplied `scripts/infer_manifest.py` command expects:

```text
checkpoints/ours_EPS-SCTA_best_checkpoint.pkl
weights/vgg_normalised.pth
```

## Verification

Run the following from the repository root after downloading the assets:

```powershell
Get-FileHash .\weight_packages\*.zip -Algorithm SHA256
python scripts/verify_project_assets.py
```

The authoritative package hashes are stored in `weight_packages/SHA256SUMS.txt` in the release asset bundle. Do not use the old exploratory or ablation checkpoints for reproducing the reported final model.

## 中文说明

大体积模型参数放在 [v1.0.0 GitHub Release](https://github.com/FishSafe1896/Edge-Palette-SCTA/releases/tag/v1.0.0) 中，不放入 Git 历史。发布包只包含最终模型和对比模型参数，不包含消融实验权重。

下载五个 ZIP 后解压到仓库根目录。最终模型的最优权重应放置为：

```text
checkpoints/ours_EPS-SCTA_best_checkpoint.pkl
weights/vgg_normalised.pth
```

StyTR-2、CAST、S2WAT 和 Aflutter Craft 的参数应分别放到各自源码目录中原脚本要求的位置。`weight_packages/SHA256SUMS.txt` 保存了发布包的 SHA-256 校验值。完成解压后运行 `python scripts/verify_project_assets.py` 检查项目资源。
