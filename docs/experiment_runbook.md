# Experiment Runbook

## Current Paper Experiment Scope

The current paper-facing proposed method is:

```text
Ours (Edge-Palette-SCTA)
```

It corresponds to the previous ablation run named `w/o frequency prior`.
The old `Full Ours` / `Lossbalanced ours` result is no longer the main method;
it is retained as the ablation variant `+ frequency prior`.

## Final Naming Map

| Paper name | Traceable source |
|---|---|
| Ours (Edge-Palette-SCTA) | `outputs/checkpoints/ablation_wo_frequency_prior_seed2026_40k/best_checkpoint.pkl` |
| Original baseline | `outputs/checkpoints/original_baseline_rawstyle_seed2026_40k/best_checkpoint.pkl` |
| StyTR-2 | `outputs/comparison/StyTR-2_style_0011_0020` |
| CAST | `outputs/comparison/CAST_pytorch_style_0011_0020` |
| S2WAT | `outputs/comparison/S2WAT_style_0011_0020` |
| + frequency prior | `outputs/checkpoints/ours_ppeb_scta_lossbalanced_seed2026_40k/best_checkpoint.pkl` |

## Paper-Ready Materials

The current paper-ready material folder is:

```text
outputs/paper_ready
```

Important files:

- `outputs/paper_ready/MATERIALS_INDEX.md`
- `outputs/paper_ready/tables/paper_main_comparison.csv`
- `outputs/paper_ready/tables/paper_main_comparison.md`
- `outputs/paper_ready/tables/paper_ablation.csv`
- `outputs/paper_ready/tables/paper_ablation.md`
- `outputs/paper_ready/tables/paper_material_source_map.csv`
- `outputs/paper_ready/figures/paper_main_comparison_sheet.jpg`
- `outputs/paper_ready/figures/paper_ablation_sheet.jpg`

## Final Paper Metrics

The final paper tables use:

- `SSIM`: content structure preservation, higher is better.
- `LPIPS-content`: perceptual distance to the content image, lower is better.
- `FID`: distribution distance to the paper-cutting style set, lower is better.
- `Color count`: dominant color compactness, lower is better for red-white paper-cutting.

`LPIPS-style` is excluded from paper-ready tables. The style image is a
reference, but this task does not require the output to be perceptually close
to one specific style image.

## Main Comparison Set

The main comparison uses:

- Manifest: `data/manifests/test_pairs_style_0011_0020.csv`
- Ours output: `outputs/inference/ablation_wo_frequency_prior_best_test_pairs_style_0011_0020`
- Original baseline output: `outputs/inference/original_baseline_rawstyle_best_test_pairs_style_0011_0020`
- External outputs:
  - `outputs/comparison/StyTR-2_style_0011_0020`
  - `outputs/comparison/CAST_pytorch_style_0011_0020`
  - `outputs/comparison/S2WAT_style_0011_0020`

## Ablation Set

The ablation table uses:

| Paper variant | Traceable old result | Meaning |
|---|---|---|
| Ours (Edge-Palette-SCTA) | `Ours w/o frequency prior` | Final model |
| w/o edge prior | `Ours w/o edge prior` | Removes edge branch and edge loss |
| w/o SCTA | `Ours w/o SCTA` | Removes the style-conditioned texture adapter |
| w/o paper-cut prior loss | `Ours w/o cut prior loss` | Removes the task-specific paper-cut prior loss |
| + frequency prior | `Full Ours` | Adds frequency branch and frequency/texture loss |

## Current Interpretation

The final method should be described as an edge-palette guided paper-cutting
style-transfer model with lightweight style conditioning. Frequency and
texture priors were implemented and tested, but they are not the main
contribution because the ablation did not improve the selected paper metrics
and may increase fragmented high-frequency artifacts.

## Keep / Delete Policy

Keep these artifacts for the current paper:

- `outputs/paper_ready`
- `outputs/checkpoints/ablation_wo_frequency_prior_seed2026_40k`
- `outputs/checkpoints/original_baseline_rawstyle_seed2026_40k`
- `outputs/checkpoints/ours_ppeb_scta_lossbalanced_seed2026_40k`
- `outputs/checkpoints/ablation_wo_edge_prior_seed2026_40k`
- `outputs/checkpoints/ablation_wo_scta_seed2026_40k`
- `outputs/checkpoints/ablation_wo_cut_loss_seed2026_40k`
- `outputs/inference/ablation_wo_frequency_prior_best_test_pairs_style_0011_0020`
- `outputs/inference/original_baseline_rawstyle_best_test_pairs_style_0011_0020`
- `outputs/inference/ours_ppeb_scta_lossbalanced_best_test_pairs_style_0011_0020`
- `outputs/inference/ablation_wo_edge_prior_best_test_pairs_style_0011_0020`
- `outputs/inference/ablation_wo_scta_best_test_pairs_style_0011_0020`
- `outputs/inference/ablation_wo_cut_loss_best_test_pairs_style_0011_0020`
- `outputs/comparison/*_style_0011_0020`
- `outputs/metrics_full_style_0011_0020`
- `outputs/metrics_ablation_style_0011_0020`

Delete only clearly abandoned smoke runs, failed intermediate runs, duplicate
loss-history copies, and deprecated experiment branches after path review.

---

# 实验运行手册

## 当前论文实验范围

当前论文中的最终方法为：

```text
Ours (Edge-Palette-SCTA)
```

它对应之前消融实验里的 `w/o frequency prior`。旧的 `Full Ours` /
`Lossbalanced ours` 不再作为主模型，而是在论文中保留为消融项：

```text
+ frequency prior
```

## 最终命名映射

| 论文名称 | 可追溯来源 |
|---|---|
| Ours (Edge-Palette-SCTA) | `outputs/checkpoints/ablation_wo_frequency_prior_seed2026_40k/best_checkpoint.pkl` |
| Original baseline | `outputs/checkpoints/original_baseline_rawstyle_seed2026_40k/best_checkpoint.pkl` |
| StyTR-2 | `outputs/comparison/StyTR-2_style_0011_0020` |
| CAST | `outputs/comparison/CAST_pytorch_style_0011_0020` |
| S2WAT | `outputs/comparison/S2WAT_style_0011_0020` |
| + frequency prior | `outputs/checkpoints/ours_ppeb_scta_lossbalanced_seed2026_40k/best_checkpoint.pkl` |

## 论文材料

当前论文可用材料统一放在：

```text
outputs/paper_ready
```

重要文件：

- `outputs/paper_ready/MATERIALS_INDEX.md`
- `outputs/paper_ready/tables/paper_main_comparison.csv`
- `outputs/paper_ready/tables/paper_main_comparison.md`
- `outputs/paper_ready/tables/paper_ablation.csv`
- `outputs/paper_ready/tables/paper_ablation.md`
- `outputs/paper_ready/tables/paper_material_source_map.csv`
- `outputs/paper_ready/figures/paper_main_comparison_sheet.jpg`
- `outputs/paper_ready/figures/paper_ablation_sheet.jpg`

## 最终论文指标

正式论文表格使用：

- `SSIM`：内容结构保持，越高越好。
- `LPIPS-content`：输出图与内容图的感知距离，越低越好。
- `FID`：输出分布与剪纸风格图集的距离，越低越好。
- `Color count`：主导颜色数量，越低越符合红白剪纸的颜色紧凑性。

`LPIPS-style` 不进入正式论文表格。style 图是参考图，但本文任务并不要求输出图与某一张 style 图在感知空间上接近。

## 主实验集合

主实验使用：

- 配对表：`data/manifests/test_pairs_style_0011_0020.csv`
- 本文方法输出：`outputs/inference/ablation_wo_frequency_prior_best_test_pairs_style_0011_0020`
- 原作者 baseline 输出：`outputs/inference/original_baseline_rawstyle_best_test_pairs_style_0011_0020`
- 外部方法输出：
  - `outputs/comparison/StyTR-2_style_0011_0020`
  - `outputs/comparison/CAST_pytorch_style_0011_0020`
  - `outputs/comparison/S2WAT_style_0011_0020`

## 消融实验集合

消融表使用：

| 论文消融名称 | 原始结果名 | 含义 |
|---|---|---|
| Ours (Edge-Palette-SCTA) | `Ours w/o frequency prior` | 最终主模型 |
| w/o edge prior | `Ours w/o edge prior` | 去掉边缘分支和边缘损失 |
| w/o SCTA | `Ours w/o SCTA` | 去掉风格条件调制模块 |
| w/o paper-cut prior loss | `Ours w/o cut prior loss` | 去掉任务特定的剪纸先验损失 |
| + frequency prior | `Full Ours` | 加入频域分支和频域/纹理损失 |

## 当前解释口径

最终方法应表述为：一种边缘-调色板引导、带轻量风格条件调制的剪纸风格迁移模型。频域/纹理先验已经实现并测试，但不作为当前主贡献，因为消融显示它没有提升选定论文指标，并且可能增加碎纹理和高频噪声。

## 保留 / 删除规则

当前论文必须保留：

- `outputs/paper_ready`
- `outputs/checkpoints/ablation_wo_frequency_prior_seed2026_40k`
- `outputs/checkpoints/original_baseline_rawstyle_seed2026_40k`
- `outputs/checkpoints/ours_ppeb_scta_lossbalanced_seed2026_40k`
- `outputs/checkpoints/ablation_wo_edge_prior_seed2026_40k`
- `outputs/checkpoints/ablation_wo_scta_seed2026_40k`
- `outputs/checkpoints/ablation_wo_cut_loss_seed2026_40k`
- `outputs/inference/ablation_wo_frequency_prior_best_test_pairs_style_0011_0020`
- `outputs/inference/original_baseline_rawstyle_best_test_pairs_style_0011_0020`
- `outputs/inference/ours_ppeb_scta_lossbalanced_best_test_pairs_style_0011_0020`
- `outputs/inference/ablation_wo_edge_prior_best_test_pairs_style_0011_0020`
- `outputs/inference/ablation_wo_scta_best_test_pairs_style_0011_0020`
- `outputs/inference/ablation_wo_cut_loss_best_test_pairs_style_0011_0020`
- `outputs/comparison/*_style_0011_0020`
- `outputs/metrics_full_style_0011_0020`
- `outputs/metrics_ablation_style_0011_0020`

只能在确认路径后删除明确废弃的 smoke run、失败中间实验、重复 loss history 副本和过期实验分支。
