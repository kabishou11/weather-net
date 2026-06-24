# Third-Party References

本项目核心训练、验证和推理代码为本仓库实现。允许复用公开依赖和公开预训练权重，不复制第三方比赛提交代码。

## 主要依赖

- `timm` / `pytorch-image-models`：用于创建 ConvNeXt、EfficientNet、Swin 等公开预训练分类模型。License: Apache-2.0。
- `PyTorch` / `torchvision`：训练与张量计算框架。License: BSD-style。
- `albumentations`：图像增强流水线。License: MIT。
- `scikit-learn`：分层 K 折与辅助指标。License: BSD-3-Clause。

## 参考但不直接复制的公开项目

- `huggingface/pytorch-image-models`：模型创建和预训练权重来源。
- 常见 GitHub 天气分类 notebook/仓库：仅参考数据分析、混淆矩阵和错误样本分析思路。

## 论文方法参考

- Focal Loss：参考 Lin et al., "Focal Loss for Dense Object Detection"，用于降低易分类样本权重，让训练更关注难样本。
- Class-Balanced Loss：参考 Cui et al., "Class-Balanced Loss Based on Effective Number of Samples"，用于按类别有效样本数调整训练权重，服务 macro F1 和长尾天气类。
- Balanced Softmax / Logit Adjustment：参考 Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition" 与 Menon et al., "Long-tail learning via logit adjustment"，本项目实现可选 `balanced_softmax` loss，按训练 fold 的类别计数修正 logits，不复制论文或第三方代码。
- LDAM：参考 Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"，本项目实现可选 `ldam` loss，按训练 fold 类别计数给少数类更大分类 margin，不改变线上推理结构。
- Decoupling / classifier re-balancing：参考 Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition" 及 `facebookresearch/classifier-balancing` 的分类头去偏思想；本项目先实现 tau-normalization checkpoint 后处理，只调整分类头权重范数，不复制第三方代码。
- ConvNeXt V2：参考 Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"，通过 `timm` 使用公开预训练骨干，不复制论文或第三方实现代码。
- Model Soups：参考 Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"，本项目增加 OOF-gated greedy 权重搜索，只对同架构 checkpoint 做参数平均。
- Bootstrap resampling：参考 Efron 的 bootstrap 重采样思想，用于 OOF 决策参数稳定性评估；本项目只用它过滤不稳定的 per-class bias，不引入额外训练数据。
- Online Hard Example Mining / hard-sample reweighting：参考 Shrivastava et al. 的难样本挖掘思想；本项目用 OOF 错分、低 margin、高 loss 生成下一轮训练的 `sample_weight`，不改变线上推理结构。
- FixMatch / Noisy Student：参考 Sohn et al. 与 Xie et al. 的高置信伪标签思路；本项目只做 OOF 估计的 class-aware 阈值和每类配额，不使用隐藏测试标签。
- AugMix：参考 Hendrycks et al., "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty"，本项目实现天气增强三视图和 JSD consistency 训练正则，不复制官方实现代码。
- CLIP / DINOv2：参考 Radford et al. 与 Oquab et al. 的通用视觉语义特征思想；本项目只消费离线导出的 image embedding 做伪标签语义原型裁判，不复制模型实现，不把额外大模型放入线上推理。

比赛提交前应确认官方规则允许使用公开预训练权重；若评测环境不能联网，应提前缓存权重或关闭 `pretrained` 并加载本地 checkpoint。
