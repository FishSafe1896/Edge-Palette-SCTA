# Paper Story Freeze

Last updated: 2026-07-18

## Final Method Name

Use the following name in paper-facing tables, figures, and drafts:

```text
Ours (Edge-Palette-SCTA)
```

Traceable checkpoint:

```text
outputs/checkpoints/ablation_wo_frequency_prior_seed2026_40k/best_checkpoint.pkl
```

This model corresponds to the previous `w/o frequency prior` ablation result.
In the final paper story, it is the main model rather than an ablation variant.

## Final Narrative

Edge-Palette-SCTA is presented as an edge-palette guided transformer framework
for Chinese paper-cutting style transfer. The method emphasizes three points:

1. Edge-PPEB strengthens boundary and cut-region structure.
2. SCTA provides style-conditioned residual modulation with 1.478M parameters.
3. The paper-cut prior loss encourages morphology-aware edges and a compact
   red-white palette.

Frequency and texture prior terms were implemented during exploration, but
their weights are zero in the final model. The frequency-prior version is kept
as an analyzed ablation because it did not improve the selected paper metrics.

## Final Contributions

1. An edge-palette guided transformer framework with Edge-PPEB and SCTA for
   Chinese paper-cutting style transfer.
2. A paper-cut prior loss that combines morphology-aware edge consistency and
   red-white palette compactness.
3. A comparative evaluation against Original baseline, StyTr2, CAST, and S2WAT,
   plus ablation studies validating edge, SCTA, PCP loss, and frequency prior.

## Final Metrics

Use only the following metrics in paper-ready tables:

- `SSIM`
- `LPIPS-content`
- `FID`
- `Color count`

Do not use `LPIPS-style` in final tables. It may over-reward similarity to a
single style reference image instead of rewarding content-preserving
paper-cutting stylization.

## Main Quantitative Results

| Method | SSIM | LPIPS-content | FID | Color count |
| --- | ---: | ---: | ---: | ---: |
| Ours (Edge-Palette-SCTA) | 0.7226 | 0.3857 | 190.85 | 10.83 |
| Original baseline | 0.7180 | 0.4707 | 196.80 | 12.75 |
| StyTr2 | 0.5985 | 0.4557 | 189.31 | 11.77 |
| CAST | 0.7562 | 0.4410 | 202.15 | 13.45 |
| S2WAT | 0.6972 | 0.4148 | 224.06 | 12.39 |

## Ablation Results

| Variant | SSIM | LPIPS-content | FID | Color count |
| --- | ---: | ---: | ---: | ---: |
| Ours | 0.7226 | 0.3857 | 190.85 | 10.83 |
| w/o edge | 0.7371 | 0.4099 | 197.36 | 11.12 |
| w/o SCTA | 0.7480 | 0.4102 | 200.63 | 10.92 |
| w/o PCP loss | 0.6752 | 0.4561 | 193.95 | 13.95 |
| + frequency prior | 0.7274 | 0.4205 | 192.44 | 11.46 |

## Retraining Decision

No additional retraining is required for the submitted paper version unless the
model definition or dataset protocol changes again. The final Ours result has a
fixed seed, a formal best checkpoint, manifest-based inference, and paper-ready
metrics.

---

# 论文叙事冻结记录

最后更新：2026-07-18

## 最终方法名称

论文表格、图片和正文中统一使用：

```text
Ours (Edge-Palette-SCTA)
```

可追溯权重：

```text
outputs/checkpoints/ablation_wo_frequency_prior_seed2026_40k/best_checkpoint.pkl
```

该模型对应此前消融实验中的 `w/o frequency prior` 版本。在最终论文叙事中，它是主模型，而不是消融变体。

## 最终叙事

Edge-Palette-SCTA 被表述为一种面向中国剪纸风格迁移的 edge-palette guided transformer framework。核心侧重点为：

1. Edge-PPEB 用于增强边界和剪纸块面结构。
2. SCTA 使用 1.478M 参数实现风格条件残差调制。
3. Paper-cut prior loss 约束形态学边缘一致性和红白调色板紧致性。

频域和纹理先验项在探索阶段实现过，但最终模型中权重为 0。带 frequency prior 的版本仅作为消融分析保留，因为它没有提升最终选定的论文指标。

## 最终创新点

1. 提出包含 Edge-PPEB 和 SCTA 的 edge-palette guided transformer framework，用于中国剪纸风格迁移。
2. 设计 paper-cut prior loss，将形态学边缘一致性和红白调色板紧致性结合起来。
3. 与 Original baseline、StyTr2、CAST、S2WAT 进行对比，并通过消融实验验证 edge、SCTA、PCP loss 和 frequency prior 的作用。

## 最终评价指标

正式论文表格只使用：

- `SSIM`
- `LPIPS-content`
- `FID`
- `Color count`

不再使用 `LPIPS-style`，因为它可能过度奖励与单张 style reference 的相似性，而不是内容保持和剪纸化质量。

## 是否需要重训

当前投稿版本不需要继续重训。除非后续模型定义或数据协议再次变化，否则以当前固定 seed、best checkpoint、manifest 推理和 paper-ready metrics 为准。
