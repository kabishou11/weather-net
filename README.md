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

使用 OOF 决策参数推理：

```bash
python3 infer.py \
  --checkpoint outputs/convnext_tiny/convnext_tiny_fold0.pt \
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
  --checkpoint outputs/convnext_tiny/convnext_tiny_fold0.pt \
  --output-dir outputs/decision/exp001
```

多模型 OOF 决策：

```bash
python3 oof_decide.py \
  --oof outputs/convnext_tiny/oof/oof_probabilities.npz outputs/fast_effnet/oof/oof_probabilities.npz \
  --checkpoint outputs/convnext_tiny/convnext_tiny_fold0.pt outputs/fast_effnet/efficientnet_b0_fold0.pt \
  --output-dir outputs/decision/exp002
```

输出的 `decision_params.json` 是提交前的冻结决策产物，避免只凭单次验证分数手填权重。

## 伪标签

对未标注图片生成高置信伪标签：

```bash
python3 pseudo_label.py \
  --checkpoint outputs/convnext_tiny/convnext_tiny_fold0.pt \
  --test-dir data/unlabeled \
  --threshold 0.95 \
  --output pseudo_labels.csv
```

`pseudo_labels.csv` 字段为 `image,label,confidence`。只建议把高置信样本合并进下一轮训练，并保留原始训练集验证划分，避免伪标签污染本地验证分数。

合并原始训练集和伪标签：

```bash
python3 merge_pseudo_labels.py \
  --train-dir data/train \
  --pseudo-csv pseudo_labels.csv \
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
