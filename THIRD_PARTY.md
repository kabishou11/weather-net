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
- ConvNeXt V2：参考 Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"，通过 `timm` 使用公开预训练骨干，不复制论文或第三方实现代码。

比赛提交前应确认官方规则允许使用公开预训练权重；若评测环境不能联网，应提前缓存权重或关闭 `pretrained` 并加载本地 checkpoint。
