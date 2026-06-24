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

cRT 分类头重训：

```bash
python3 classifier_rebalance.py \
  --method crt \
  --checkpoint outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0.pt \
  --train-csv data/train.csv \
  --image-root data/images \
  --output outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0_crt.pt \
  --epochs 4 \
  --lr 0.001 \
  --sampler-mode sqrt
```

`--method crt` 会加载已有 checkpoint，冻结 backbone，只训练分类头参数；训练数据会强制沿用 checkpoint 内的 `class_to_idx`，避免重新读 CSV 时类别顺序变化导致标签错位。默认用 `sqrt` 采样在类均衡和原分布之间折中，也可用 `class_balanced` 做更强尾类纠偏。该步骤训练成本很低、推理成本不变，适合在强 backbone 收敛后修正长尾类别决策边界。建议先对单 fold 或 OOF 最差类别对应 fold 快筛，再扩展到所有 fold；若大类 precision 明显下降，则回退到原 checkpoint 或 tau-norm。

LWS 分类头缩放：

```bash
python3 classifier_rebalance.py \
  --method lws \
  --checkpoint outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0.pt \
  --train-csv data/train.csv \
  --image-root data/images \
  --output outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0_lws.pt \
  --epochs 4 \
  --lr 0.001 \
  --sampler-mode sqrt
```

`--method lws` 固定 backbone 和原分类头方向，只学习每个类别的 logit scale，然后把 scale 折叠回 classifier weight/bias，生成普通 checkpoint，推理成本为零。它用于修正长尾类的分类边界，表达力高于 tau-normalization，但也更容易在小验证集上过拟合；默认 scale 限制在 `[0.25, 4.0]`，并会拒绝非 finite scale 或非法 `class_to_idx`。不要用伪标签训练 LWS/cRT，也不要只看训练 loss 下降；必须用 OOF 或本地验证集检查 macro F1、每类 F1 和头部类 precision。

自动生成并汇总分类头重平衡网格：

```bash
python3 classifier_rebalance_grid.py \
  --checkpoint outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0.pt \
  --output-dir outputs/convnextv2_384/rebalance_grid/fold0 \
  --tau 0.25 0.5 0.75 1.0 \
  --crt-sampler-mode sqrt class_balanced \
  --lws-sampler-mode sqrt class_balanced \
  --train-csv data/train.csv \
  --image-root data/images \
  --run
```

跑完候选后，用 `validate.py --output-json` 分别得到 baseline 和候选 metrics，再汇总：

```bash
python3 classifier_rebalance_grid.py \
  --checkpoint outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0.pt \
  --output-dir outputs/convnextv2_384/rebalance_grid/fold0 \
  --tau 0.25 0.5 0.75 1.0 \
  --crt-sampler-mode sqrt class_balanced \
  --baseline-metrics outputs/convnextv2_384/rebalance_grid/fold0/baseline_metrics.json \
  --candidate-metrics outputs/convnextv2_384/rebalance_grid/fold0/tau0p25_metrics.json outputs/convnextv2_384/rebalance_grid/fold0/tau0p50_metrics.json outputs/convnextv2_384/rebalance_grid/fold0/tau0p75_metrics.json outputs/convnextv2_384/rebalance_grid/fold0/tau1p00_metrics.json outputs/convnextv2_384/rebalance_grid/fold0/crt_sqrt_metrics.json outputs/convnextv2_384/rebalance_grid/fold0/crt_class_balanced_metrics.json \
  --min-delta-macro-f1 0.003 \
  --min-per-class-f1 0.6 \
  --tie-epsilon 0.001
```

如果验证集已经准备好，也可以让 grid runner 自动验证 baseline 和所有候选，并直接输出 summary：

```bash
python3 classifier_rebalance_grid.py \
  --checkpoint outputs/convnextv2_384/convnextv2_tiny.fcmae_ft_in22k_in1k_fold0.pt \
  --output-dir outputs/convnextv2_384/rebalance_grid/fold0 \
  --tau 0.25 0.5 0.75 1.0 \
  --crt-sampler-mode sqrt class_balanced \
  --lws-sampler-mode sqrt class_balanced \
  --train-csv data/train.csv \
  --image-root data/images \
  --run \
  --validate \
  --val-csv data/val.csv \
  --val-image-root data/images \
  --val-batch-size 64 \
  --min-delta-macro-f1 0.003 \
  --min-per-class-f1 0.6 \
  --tie-epsilon 0.001
```

`--validate` 会写出 `baseline_metrics.json` 和每个候选的 `{candidate}_metrics.json`，再调用同一套门控汇总逻辑。它不能和 `--baseline-metrics/--candidate-metrics` 混用，避免自动验证结果被手工 metrics 路径误覆盖。

汇总会输出 `classifier_rebalance_grid_summary.json/csv`，只有 macro F1 达到增益门槛且最低类 F1 不崩的候选才会被选中。这个门控用于避免在小验证集上被单一高分候选骗过去。

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

如果 `pseudo_labels_filtered.csv` 已包含 `teacher_{class}` soft label，可显式启用 Noisy Student 式伪标签蒸馏合并：

```bash
python3 merge_pseudo_labels.py \
  --train-csv data/train_teacher.csv \
  --image-root data/images \
  --pseudo-csv pseudo_labels_filtered.csv \
  --pseudo-image-root data/unlabeled \
  --min-confidence 0.95 \
  --allow-pseudo-teacher \
  --output data/merged_teacher_pseudo.csv
```

启用后会要求伪标签 teacher 概率覆盖全部类别、非负有限、和为 1，且 teacher top-1 必须和伪标签类别一致；真实标注行若已有 OOF soft teacher 会原样保留，否则补 one-hot teacher，保证蒸馏训练每行都有完整 soft target。

用合并后的 CSV 再训练：

```bash
python3 train.py \
  --config configs/convnext_tiny.yaml \
  --train-csv merged_train.csv \
  --epochs 5 \
  --output-dir outputs/convnext_pseudo
```

## OOF teacher distillation

训练 CSV 可选加入按类别命名的 teacher soft label 列：

```csv
image,label,teacher_cloudy,teacher_rain,teacher_shine,teacher_sunrise
0001.jpg,rain,0.02,0.94,0.03,0.01
```

列名必须是 `teacher_{class_name}`，且覆盖 `class_to_idx` 中的每个类别；每行概率必须非负、有限、和为 1。建议 teacher 只来自 OOF 模型、不同架构 ensemble 或独立 teacher，不能用同一 fold 的学生模型预测自己的训练样本，否则本地 F1 会被泄漏污染。

如果已经跑完 K 折并生成 `oof_probabilities.npz`，可直接把 OOF soft label 写回训练 CSV：

```bash
python3 teacher_from_oof.py \
  --train-csv data/train.csv \
  --image-root data/images \
  --oof outputs/convnextv2_384/oof/oof_probabilities.npz \
  --max-top1-mismatch-rate 0.25 \
  --min-mean-true-probability 0.65 \
  --min-samples-per-class 2 \
  --output data/train_teacher.csv
```

该工具会要求 OOF 覆盖每个真实标注样本、`class_names` 顺序与训练 CSV 推断类别顺序一致、概率合法且 OOF 来源全为 `labeled`。建议同时打开 teacher 质量门控：限制 top-1 与真标签冲突率、要求平均 true-class probability 达标，并确认每类有足够 OOF 覆盖。写出后建议先跑 preflight：

```bash
python3 training_preflight.py \
  --config configs/convnextv2_384.yaml \
  --train-csv data/train_teacher.csv \
  --image-root data/images \
  --check-image-exists \
  --check-unique-image-id
```

配置示例：

```yaml
train:
  distillation_alpha: 0.3
  distillation_temperature: 2.0
```

`distillation_alpha` 控制监督标签和 teacher KL 的混合比例。该路径支持 MixUp/CutMix：训练时会用同一个 `lam/permutation` 同步混合 teacher soft label 和样本权重，避免“图像混了但软标签没混”的目标错位。若使用 `augmix_jsd`，teacher KL 只作用在 clean view，JSD consistency 仍约束 clean/aug views。

## 外部公开数据

已验证可下载的数据源：

- MWD / Multi-class Weather Dataset：Kaggle `pratik2901/multiclass-weather-dataset`，1125 张，Kaggle 标 CC BY 4.0，类别映射后为 `cloudy/rain/sunny`。
- WEAPD mirror / Weather Image Recognition：Kaggle `jehanbhathena/weather-dataset`，6862 张，Kaggle 标 CC0，类别映射后为 `dew/fog/lightning/rain/rainbow/sandstorm/snow`。

下载示例：

```bash
mkdir -p data/external/mwd_kaggle data/external/weapd_kaggle
curl -L --fail --retry 3 \
  -o data/external/mwd_kaggle/multiclass-weather-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/pratik2901/multiclass-weather-dataset
curl -L --fail --retry 3 \
  -o data/external/weapd_kaggle/weather-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/jehanbhathena/weather-dataset
unzip -q -o data/external/mwd_kaggle/multiclass-weather-dataset.zip -d data/external/mwd_kaggle
unzip -q -o data/external/weapd_kaggle/weather-dataset.zip -d data/external/weapd_kaggle
```

生成外部 CSV：

```bash
python3 external_dataset_manifest.py \
  --image-root "data/external/mwd_kaggle/Multi-class Weather Dataset" \
  --dataset-name external_mwd \
  --dataset-url https://data.mendeley.com/datasets/4drtyfjtfy/1 \
  --license "CC BY 4.0" \
  --doi 10.17632/4drtyfjtfy.1 \
  --label-map-preset mwd \
  --class-map outputs/convnextv2_384/class_to_idx.json \
  --output-csv data/external/mwd_kaggle/external_train.csv \
  --summary-json data/external/mwd_kaggle/summary.json

python3 external_dataset_manifest.py \
  --image-root data/external/weapd_kaggle/dataset \
  --dataset-name external_weapd \
  --dataset-url "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/M8JQCR" \
  --license CC0 \
  --doi 10.7910/DVN/M8JQCR \
  --label-map-preset weapd \
  --class-map outputs/convnextv2_384/class_to_idx.json \
  --output-csv data/external/weapd_kaggle/external_train.csv \
  --summary-json data/external/weapd_kaggle/summary.json
```

如果官方类别不包含 `dew/lightning/rainbow/sandstorm` 这类标签，`--class-map` 会直接阻止它们进入 manifest；可用自定义 JSON 映射并开启 `--drop-unmapped`，只保留能对齐官方类别的样本。summary 会记录 `dataset_url/license/doi/label_map_sha256`，便于赛前审计数据来源。

外部数据默认不直接等权并入官方训练集。推荐用法是：先按官方类别过滤，再加 `source=external` 和低 `sample_weight`，或只导出 DINOv2/CLIP/timm embedding 给 `embedding_guard.py` 做伪标签语义裁判。若比赛规则禁止外部数据，则这些数据只能用于本地鲁棒性分析，不进入最终训练。

## 服务器训练前 preflight

正式上 4090/服务器前先跑训练门控，避免一次长训练被 CSV、类别映射、外部数据、teacher 列或推理预算问题毁掉：

```bash
python3 training_preflight.py \
  --config configs/convnextv2_384.yaml \
  --train-csv data/train.csv \
  --image-root data/images \
  --class-map outputs/convnextv2_384/class_to_idx.json \
  --check-image-exists \
  --check-unique-image-id \
  --check-unique-realpath \
  --min-images-per-class 2 \
  --output outputs/preflight_train.json
```

如果训练 CSV 包含 `source=external` 或 `external_*`，必须显式声明规则允许外部公开数据：

```bash
python3 training_preflight.py \
  --config configs/convnextv2_384.yaml \
  --train-csv data/train_with_external.csv \
  --image-root data/images \
  --allow-external-data
```

如果训练 CSV 包含 `source=pseudo`，建议把伪标签置信度和数量比例设成硬门槛：

```bash
python3 training_preflight.py \
  --config configs/convnextv2_384.yaml \
  --train-csv data/merged_train.csv \
  --image-root data/images \
  --pseudo-min-confidence 0.95 \
  --pseudo-max-ratio 0.5
```

如果已经有 `infer.py` 生成的 `.stats.json`，可把推理时间也纳入同一个 gate：

```bash
python3 training_preflight.py \
  --config configs/convnextv2_384.yaml \
  --train-csv data/train.csv \
  --image-root data/images \
  --inference-stats outputs/submission.stats.json \
  --max-seconds-per-image 0.05 \
  --max-checkpoints 1
```

当 `train.distillation_alpha > 0` 时，preflight 会要求每一行都有完整 `teacher_{class_name}` 概率列，并默认要求 teacher 训练 CSV 只包含真实标注行。若要训练 Noisy Student 伪标签 teacher CSV，必须显式加 `--allow-pseudo-teacher-distillation`，并同时设置 `--pseudo-min-confidence` 与 `--pseudo-max-ratio`。`--train-csv` 和 `--train-dir` 不能同时设置；当外部数据未显式允许、伪标签低于门槛、某类样本低于门槛、推理 stats 超预算或 TTA 未被允许时，会直接失败。

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
