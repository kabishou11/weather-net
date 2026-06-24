# Weather Net

天气图片分类比赛工程，目标是快速得到高 F1 且可在线评测的训练/推理流水线。

## 安装

```bash
python3 -m pip install -r requirements-dev.txt
```

## 数据格式

支持两种训练数据格式：

```text
data/train/
  rain/xxx.jpg
  sunny/yyy.jpg
```

或 CSV：

```csv
image,label
xxx.jpg,rain
yyy.jpg,sunny
```

## 训练

```bash
python3 train.py \
  --config configs/convnext_tiny.yaml \
  --train-dir data/train \
  --epochs 10 \
  --batch-size 32 \
  --output-dir outputs/convnext_tiny
```

CSV 训练：

```bash
python3 train.py \
  --train-csv data/train.csv \
  --image-root data/images \
  --config configs/convnext_tiny.yaml
```

高分辨率强单模型基线：

```bash
python3 train.py \
  --train-csv data/train.csv \
  --image-root data/images \
  --config configs/convnextv2_384.yaml
```

`convnextv2_384.yaml` 面向 24G 4090 设计，主线是 ConvNeXtV2 预训练骨干、384 输入、WeatherAugMix、EMA、MixUp/CutMix，以及 `class_balanced_focal` 长尾损失。这个组合对齐 macro F1：少数天气类会获得更高训练权重，focal 项会让模型更关注难样本。不要只看单次验证分数，最终以 5-fold OOF macro F1、每类 F1 和混淆矩阵判断是否保留。

长尾先验校正对照：

```bash
python3 train.py \
  --train-csv data/train.csv \
  --image-root data/images \
  --config configs/convnextv2_384_balanced_softmax.yaml
```

`convnextv2_384_balanced_softmax.yaml` 使用 Balanced Softmax，在训练 loss 内按当前 fold 的训练集类别计数调整 logits。它适合天气类别明显不均衡、线上评分看 macro F1 的场景；默认 `sampler_mode: auto` 不再额外做类均衡采样，避免“先验校正 + 重采样”双重放大尾类。建议先跑 1 fold 或 3 fold 和 `class_balanced_focal` 对比，若少数类 F1 上升且大类 precision 没崩，再进入 5 fold/soup。

长尾 margin 对照：

```bash
python3 train.py \
  --train-csv data/train.csv \
  --image-root data/images \
  --config configs/convnextv2_384_ldam.yaml
```

`convnextv2_384_ldam.yaml` 使用 LDAM loss，根据当前 fold 的训练集类别计数给少数类更大的分类 margin。它适合“少数天气类召回不足、但模型已经能学到可分特征”的阶段，推理成本为零。LDAM 默认不叠加 class-balanced 权重或自动类均衡采样，避免同时改 loss margin 和样本分布；建议和 Balanced Softmax 做平行 A/B，而不是把两者混合到同一次训练。

鲁棒性增强路线：

```bash
python3 train.py \
  --train-csv data/train.csv \
  --image-root data/images \
  --config configs/convnextv2_384_augmix_jsd.yaml
```

`convnextv2_384_augmix_jsd.yaml` 使用 `augment_policy: augmix_jsd`：每张训练图生成 clean/aug1/aug2 三视图，监督 loss 仍使用 `class_balanced_focal` 和样本权重，额外加入 AugMix 风格 JSD consistency loss。它针对雨雾、低照、眩光、压缩噪声等分布偏移，训练约增加到 3 路前向成本，但导出的 checkpoint 和普通模型一样，推理不增加耗时。该策略和 MixUp/CutMix 互斥，避免把混合样本分布拿去做一致性约束。

## 训练预算

单卡 24G RTX 4090 足够跑本工程的主路线。经验估算：

- `convnext_tiny` 224 输入、batch 32：每 1 万张图、10 epoch，约 20-60 分钟；3 折约 1-3 小时，5 折约 2-5 小时。
- `convnextv2_384` 384 输入、batch 24：每 1 万张图、18 epoch，约 1.5-4 小时；5 折约 8-20 小时。
- `convnextv2_384_augmix_jsd` 384 输入、batch 16：每 1 万张图、18 epoch，约 3-9 小时；5 折约 15-45 小时。若时间紧，先跑 3 折或只对最佳 fold/seed 做对照。
- `efficientnet_b0` 快速兜底：每 1 万张图、8 epoch，约 10-30 分钟，适合提交前压推理时间。

如果显存紧张，优先把 `batch_size` 下调到 16 或 12，保持 384 输入和预训练骨干；如果训练时间紧，优先跑 3 折 OOF，再用最佳配置补 5 折。推理默认使用 fp32 以保持概率和伪标签阈值稳定；确认本地/线上 F1 不受影响后，可在配置中设置 `infer.amp: true` 或命令行加 `--amp` 换取速度。

## 推理

```bash
python3 infer.py \
  --checkpoint outputs/convnext_tiny/convnext_tiny_fold0.pt \
  --test-dir data/test \
  --output submission.csv
```

多 checkpoint 集成：

```bash
python3 infer.py \
  --checkpoint outputs/model_a.pt outputs/model_b.pt \
  --weights 0.6 0.4 \
  --test-dir data/test \
  --tta \
  --output submission.csv
```

`--weights` 会对 logits 做归一化加权；不传时所有 checkpoint 等权平均。

提交前检查推理预算：

```bash
python3 inference_budget.py \
  --stats submission.stats.json \
  --max-seconds-per-image 0.05 \
  --max-checkpoints 1
```

`infer.py` 会在输出 CSV 同目录写出 `.stats.json`，其中包含总耗时、单张吞吐、checkpoint 数、TTA 和 AMP 状态。`inference_budget.py` 用它做提交前门控：若超出单张耗时、checkpoint 数或 TTA 预算，会返回非零退出码。省赛同分按推理时间排序时，优先提交通过该门控的 soup 单模型或快速单模型。

使用 OOF 决策参数推理：

```bash
python3 infer.py \
  --checkpoint outputs/convnext_tiny/convnext_tiny_fold0.pt outputs/convnext_tiny/convnext_tiny_fold1.pt outputs/convnext_tiny/convnext_tiny_fold2.pt \
  --test-dir data/test \
  --decision-params outputs/decision/exp001/decision_params.json \
  --sample-submission data/sample_submission.csv \
  --output submission_exp001.csv
```

`decision_params.json` 会在 logits softmax 前应用 temperature 和 per-class bias；如果文件内含 `weights` 且命令行没有显式传 `--weights`，推理会使用该权重。

## OOF 决策层

训练会在 `output_dir/oof/` 下写出：

```text
oof_predictions.csv
oof_probabilities.npz
oof_metrics.json
```

基于 OOF 搜索 ensemble 权重、temperature 和 per-class bias：

```bash
python3 oof_decide.py \
  --oof outputs/convnext_tiny/oof/oof_probabilities.npz \
  --checkpoint outputs/convnext_tiny/convnext_tiny_fold0.pt outputs/convnext_tiny/convnext_tiny_fold1.pt outputs/convnext_tiny/convnext_tiny_fold2.pt \
  --bootstrap-rounds 200 \
  --bootstrap-sample-fraction 0.8 \
  --bootstrap-min-delta-q05 0.0 \
  --output-dir outputs/decision/exp001
```

多模型 OOF 决策：

```bash
python3 oof_decide.py \
  --oof outputs/convnext_tiny/oof/oof_probabilities.npz outputs/fast_effnet/oof/oof_probabilities.npz \
  --checkpoint outputs/convnext_tiny/convnext_tiny_fold0.pt outputs/convnext_tiny/convnext_tiny_fold1.pt outputs/convnext_tiny/convnext_tiny_fold2.pt outputs/fast_effnet/efficientnet_b0_fold0.pt outputs/fast_effnet/efficientnet_b0_fold1.pt outputs/fast_effnet/efficientnet_b0_fold2.pt \
  --output-dir outputs/decision/exp002
```

输出的 `decision_params.json` 是提交前的冻结决策产物，避免只凭单次验证分数手填权重。若 `oof_probabilities.npz` 来自 K 折训练，文件内会记录每条 OOF 预测对应的 fold checkpoint；`oof_decide.py` 会把 OOF 组权重自动展开为每个 fold checkpoint 的推理权重。线上推理时请传入同一组 fold checkpoint，顺序需与 `decision_params.json` 绑定一致。

`--bootstrap-rounds` 会对 OOF 样本做分层有放回重采样，评估 per-class bias 相对 temperature-only 结果的稳定增益；若 `delta_q05_macro_f1` 低于 `--bootstrap-min-delta-q05`，会自动回退 bias，避免把 OOF 偶然性写进提交参数。这个步骤只影响决策参数搜索，不增加线上推理成本；不传 bootstrap 参数时保持旧行为，不启用门控。

基于 OOF 生成下一轮训练的 hard-sample 权重：

```bash
python3 hard_mining.py \
  --train-csv data/train.csv \
  --oof-csv outputs/convnextv2_384/oof/oof_predictions.csv \
  --error-boost 1.0 \
  --low-margin-boost 0.5 \
  --high-loss-boost 0.5 \
  --low-margin-threshold 0.1 \
  --high-loss-quantile 0.75 \
  --max-weight 2.5 \
  --output data/train_hard_weighted.csv \
  --hard-output outputs/convnextv2_384/hard_samples.csv
```

`hard_mining.py` 只给真实标注样本写回 `sample_weight`，不会用 OOF 结果改伪标签行；`hard_samples.csv` 可用于检查错分、低 margin 和高 loss 样本。该步骤适合在强单模 OOF 后做第二轮 fine-tune，目标是把训练容量集中到拖 macro F1 的混淆类和边界样本上，不增加推理成本。若已生成 `merged_train.csv`，也可以把它作为 `--train-csv` 输入，脚本会保留其中伪标签行的原始权重。

把 OOF 混淆对 hard fine-tune 做成可复现 A/B：

```bash
python3 confusion_hard_ablation.py \
  --base-config configs/convnextv2_384_hard_finetune.yaml \
  --train-csv data/train.csv \
  --oof-csv outputs/convnextv2_384/oof/oof_predictions.csv \
  --pair-confusion-boost 0.0 0.4 0.8 \
  --pair-min-support 2 \
  --pair-min-error-share 0.35 \
  --max-weight 2.2 \
  --folds 3 \
  --epochs 4 \
  --min-delta-macro-f1 0.003 \
  --min-per-class-f1 0.6 \
  --run \
  --summarize
```

这会为每个 `pair_confusion_boost` 生成 `train_confusion_hard_weighted.csv`、`confusion_hard_samples.csv`、训练配置和 `confusion_hard_ablation_summary.json`。默认 baseline 是 `pair0p00`，只有 macro F1 达到门槛且尾类 F1 不崩的 pair boost 才会被选中。这个步骤只改变训练采样，不改变推理图；推荐先 3-fold 快筛，再把最稳参数扩到 5-fold。

## Model Soup

同架构 checkpoint 可以做权重平均，得到单 checkpoint 推理包：

```bash
python3 soup_checkpoints.py \
  --checkpoints outputs/seed1/convnextv2.pt outputs/seed2/convnextv2.pt \
  --oof outputs/seed1/oof/oof_probabilities.npz outputs/seed2/oof/oof_probabilities.npz \
  --min-delta 0.0 \
  --output outputs/soup/convnextv2_oof_gated_soup.pt
```

传入 `--oof` 时会先用 OOF logits 做 greedy 权重搜索，只保留不低于最佳单 checkpoint macro F1 的 soup 权重，再平均模型参数。这个 OOF 分数用于 gate logits 权重，不等价于 soup checkpoint 的最终验证分数；生成后建议再用 `validate.py` 跑一次本地验证。这个步骤适合同一架构、同类别映射、同输入尺寸的多 seed 或 late checkpoint；不要跨架构 soup，跨架构请继续用 logits ensemble 或 OOF 决策层。

分类头 tau-normalization 去偏：

```bash
python3 classifier_rebalance.py \
  --checkpoint outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0.pt \
  --output outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0_tau.pt \
  --tau 0.5
```

`classifier_rebalance.py` 会只调整 checkpoint 中 `classifier/head/fc.weight` 每个类别权重向量的范数，不改变权重方向，也不改变模型结构。它用于快速 A/B 长尾分类头去偏，推理成本为零；建议用 `tau=0.25/0.5/0.75/1.0` 网格在 OOF 或本地验证集上筛选，只有 macro F1 和低 F1 类别同时改善时再替换提交 checkpoint。若 checkpoint 中存在多个分类头候选，脚本会要求显式传 `--head-key`，避免自动改错头。

## 伪标签

对未标注图片生成高置信伪标签：

```bash
python3 pseudo_label.py \
  --checkpoint outputs/convnext_tiny/convnext_tiny_fold0.pt \
  --test-dir data/unlabeled \
  --threshold 0.95 \
  --output pseudo_labels.csv
```

从 OOF 估计每类阈值，减少全局阈值让易分类大类吞掉伪标签：

```bash
python3 pseudo_label.py \
  --estimate-thresholds-from-oof outputs/convnextv2_384/oof/oof_probabilities.npz \
  --target-precision 0.95 \
  --min-class-threshold 0.8 \
  --fallback-class-threshold 0.99 \
  --threshold-output outputs/convnextv2_384/per_class_thresholds.json
```

使用每类阈值和每类上限生成伪标签：

```bash
python3 pseudo_label.py \
  --checkpoint outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0.pt \
  --test-dir data/unlabeled \
  --threshold 0.95 \
  --per-class-thresholds outputs/convnextv2_384/per_class_thresholds.json \
  --per-class-max-count configs/pseudo_caps.json \
  --output pseudo_labels.csv
```

`pseudo_labels.csv` 字段为 `image,label,confidence`。合并后的训练 CSV 会写出 `image,label,source,confidence,sample_weight`；伪标签权重默认为 `0.3 + 0.7 * confidence`，且会跳过与原始有标注集路径重复的伪标签、拒绝原始类别集合之外的伪标签，避免验证泄漏和污染类别空间。只建议把高置信样本合并进下一轮训练，并保留原始训练集验证划分，避免伪标签污染本地验证分数。

可选：用 CLIP/DINOv2/timm 等离线导出的 image embedding 做语义原型裁判，过滤“分类模型高置信但视觉语义不像该类”的伪标签：

```bash
python3 embedding_guard.py \
  --train-csv data/train.csv \
  --image-root data/images \
  --pseudo-csv pseudo_labels.csv \
  --embeddings outputs/embeddings/clip_or_dinov2_embeddings.npz \
  --min-similarity 0.8 \
  --min-margin 0.1 \
  --output pseudo_labels_filtered.csv \
  --rejected-output pseudo_labels_rejected.csv
```

`embeddings.npz` 需要包含 `image_id` 和 `embedding` 两个数组；`image_id` 必须覆盖训练 CSV/ImageFolder 的原始 image id 以及伪标签 CSV 的 `image` 列。该步骤只清洗伪标签，不进入线上推理；建议先审查 `pseudo_labels_rejected.csv`，确认过滤掉的是跨类污染或异常图，再把 `pseudo_labels_filtered.csv` 交给合并脚本。

合并原始训练集和伪标签：

```bash
python3 merge_pseudo_labels.py \
  --train-dir data/train \
  --pseudo-csv pseudo_labels_filtered.csv \
  --pseudo-image-root data/unlabeled \
  --min-confidence 0.95 \
  --output merged_train.csv
```

用合并后的 CSV 再训练：

```bash
python3 train.py \
  --config configs/convnext_tiny.yaml \
  --train-csv merged_train.csv \
  --epochs 5 \
  --output-dir outputs/convnext_pseudo
```

## 四天冲分顺序

1. 先用 `convnext_tiny` 跑通单模型，保存 `class_to_idx.json` 和 best checkpoint。
2. 尝试 `efficientnet_b3` 或 `swin_tiny`，保留验证集 macro F1 最优模型。
3. 用 `--folds 3` 或配置 `folds: 5` 做分层 K 折。
4. 最终根据在线推理时间选择单模型、2 模型集成或 `configs/fast_effnet.yaml` 快速模型。
5. 查看 `training_summary.json` 的每类 F1 和混淆矩阵，优先针对低 F1 类别补数据增强或调权重。
6. 用高置信伪标签做最后一轮精调，阈值从 `0.95` 起步，宁缺毋滥。

## 本地验证

```bash
python3 -m pytest tests -q
```
