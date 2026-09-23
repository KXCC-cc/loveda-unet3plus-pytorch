# ResNet34 LoveDA 受控消融计划

## 目的

以已完成的 ResNet34 U-Net 3+ 实验为固定基线，逐项评估训练策略对 LoveDA 七分类语义分割性能的影响。基线 Full-Val mIoU 参考值约为 **0.4669**。所有实验使用同一个完整 Validation Set，继续按全验证集累计混淆矩阵计算 mIoU、mean Dice 和各类别 IoU。

## 固定条件

所有实验保持以下条件不变：

- ResNet34 ImageNet IMAGENET1K_V1 encoder
- U-Net 3+ full-scale skip connections 与 deep supervision
- 输入裁剪 512 × 512
- cat_channels=64
- physical batch 1，gradient accumulation 4，effective batch 4
- SGD，momentum 0.9，weight decay 1e-4
- Poly scheduler，power 0.9，15000 optimizer updates
- AMP 开启，encoder BN 冻结
- LoveDA 标签映射、ignore_index=255 和 mIoU 定义不变

## 实验矩阵

| ID | 配置 | 唯一策略变化 | 目的 |
| --- | --- | --- | --- |
| B0 | loveda_4060_resnet34.yaml | 无 | 固定 baseline，Full-Val mIoU ≈ 0.4669 |
| B1 | resnet_pretrained_512_multiscale.yaml | 训练尺度改为 0.5、0.75、1.0、1.25、1.5、1.75 | 检查官方多尺度训练是否改善泛化 |
| B2 | resnet34_512_classaware_only.yaml | class-aware crop probability 0.5 | 单独评估类别感知裁剪 |
| B3 | resnet34_512_diff_lr.yaml | encoder LR 0.0005，decoder LR 0.0025 | 单独评估预训练编码器的较小学习率 |
| B4 | resnet34_512_focal.yaml | CE 0.5 + Dice 1.0 + Focal 0.5 | 单独评估 Focal 对难分类像素的影响 |
| B5 | resnet34_512_multiscale_classaware.yaml | Multi-scale + Class-aware | 组合实验，不作为单变量 class-aware 结果 |

原有 resnet_pretrained_512_classaware.yaml 本身继承 multi-scale 配置，因此它表示 B5 组合实验。新文件 resnet34_512_multiscale_classaware.yaml 是语义更明确的兼容别名，原配置继续保留。

## 评价与比较

每次实验完成后记录并横向比较：

1. 完整 Validation Set 的 7 类 mIoU；
2. mean Dice；
3. background、building、road、water、barren、forest、agricultural 各类别 IoU；
4. confusion matrix；
5. 最佳 checkpoint 对应的 epoch 与 optimizer step。

先比较单变量 B1～B4 与 B0。只有单变量结果明确改善后，再使用 B5 检查组合策略，避免无法判断提升来源。

## 运行顺序

推荐先运行 **B3 Differential LR only**。它不增加数据处理开销，且直接针对 ImageNet 预训练 encoder 与随机初始化 decoder 的优化速度差异。随后依次运行 B2、B1、B4。B5 应放在确认单项策略有效之后。

## 已完成结果

### B3 Differential LR only

- 状态：完成，15000 optimizer updates
- 最佳/最终 epoch：24
- Full-Val mIoU：0.444891
- mean Dice：0.608513
- val main loss：1.753863
- 相对 B0 baseline 0.466931：下降 0.022041（约 2.20 个百分点）
- 结论：当前 encoder LR 0.0005、decoder LR 0.0025 的差分学习率未提升 baseline，不建议作为最终主模型配置。

完整逐轮指标和混淆矩阵保存在 results/resnet34_512_diff_lr/。权重、预测图片和数据集不进入 Git 仓库。
