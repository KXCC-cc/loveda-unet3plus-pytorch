# LoveDA U-Net 3+：从 PNG 到分割训练

这是用于学习、生产实习和后续 LoveDA 七类训练的 PyTorch 项目。代码保留显式的全尺度 Decoder，采用原分辨率裁块训练、五级深监督、CUDA/AMP 和整图滑窗验证。

**本次只做源码/目录阅读、少量 PNG 元信息检查和文件编辑。没有执行模型 forward、训练、结构测试、benchmark、类别统计或依赖安装。下面的命令供用户以后手动执行，不代表已经验证通过。**

## 1. 本地实际检查结果

2026-09-14 检查的是 `D:/unet3` 的实际文件，不是按目录示例推测。

| Split | Rural images / masks | Urban images / masks | 图像总数 |
| --- | --- | --- | --- |
| Train | 1366 / 1366 | 1156 / 1156 | 2522 |
| Val | 992 / 992 | 677 / 677 | 1669 |
| Test | 976 / 无 mask 目录 | 820 / 无 mask 目录 | 1796 |

Train/Val 四个有标签目录按同名 PNG 配对，均无缺失或多余 mask。每个图像/标签目录抽查两个 PNG，共 20 个文件的 IHDR；抽查结果均为 1024×1024、8 bit，图像为 RGB，mask 为灰度。这里只读取文件头，没有扫描所有像素，不能据此宣称所有文件无损坏、标签值已全部验证或类别频率已知。

原项目的 full-scale 跳跃连接、PNG 配对、ignore 标签映射及混淆矩阵主体正确。主要不足是整图直接缩小、没有同步增强/AMP、缺少 E5 深监督头、checkpoint 未保存 scaler、`blocks.py` 为空、设备和标签规则分散。

## 2. 项目目录与文件职责

```text
/home/kxcc/unet3/
├─ models/
│  ├─ __init__.py       # 导出 UNet3Plus 和模型版本
│  ├─ blocks.py         # ConvBNReLU、DoubleConv
│  └─ unet3plus.py      # 显式 Encoder / Decoder / 五个输出头
├─ utils/
│  ├─ __init__.py       # 工具包声明
│  ├─ constants.py      # 类别、ignore、归一化、标签映射唯一来源
│  ├─ transforms.py     # 同步 crop / resize / 翻转 / 旋转 / 图像归一化
│  ├─ dataset.py        # PNG 配对、读取、调用变换、返回 Tensor
│  ├─ losses.py         # CE、Dice、可选 Focal、深监督加权
│  ├─ metrics.py        # 累计混淆矩阵及 IoU / Dice
│  ├─ device.py         # CUDA 选择、AMP 上下文、GradScaler 兼容
│  ├─ checkpoint.py     # 保存、恢复、主进程随机状态
│  ├─ inference.py      # 验证与推理共用的整图滑窗 logits 融合
│  └─ experiment.py     # CSV、配置、曲线、混淆矩阵、验证样本可视化
├─ dataset/             # 只读原始数据
├─ checkpoints/         # 用户训练后保存模型
├─ train.py             # 参数、DataLoader、训练和验证
├─ predict.py           # 单张图像推理及 PNG 输出
├─ test_model.py        # 用户手动进行 CPU/CUDA shape 检查
├─ compute_class_stats.py  # 用户手动统计原始标签分布
├─ requirements.txt
└─ README.md
```

建议阅读顺序：`constants.py → transforms.py → dataset.py → blocks.py → unet3plus.py → losses.py → train.py → inference.py → metrics.py → checkpoint.py → predict.py`。

## 3. LoveDA 数据结构与七个类别

```text
dataset/
├─ Train/
│  ├─ Rural/{images_png,masks_png}/
│  └─ Urban/{images_png,masks_png}/
├─ Val/
│  ├─ Rural/{images_png,masks_png}/
│  └─ Urban/{images_png,masks_png}/
└─ Test/
   ├─ Rural/images_png/
   └─ Urban/images_png/
```

同一地域内，image 和 mask 文件名必须完全相同，原始宽高必须一致。Train/Val 有标签；Test 无标签，不能计算测试 Dice/mIoU。[LoveDA 官方项目](https://github.com/Junjue-Wang/LoveDA)也明确说明七个类别和 no-data 规则。

| 原始值 | 训练值 | 类别 |
| --- | --- | --- |
| 0 | 255，ignore | no-data |
| 1 | 0 | background，背景 |
| 2 | 1 | building，建筑 |
| 3 | 2 | road，道路 |
| 4 | 3 | water，水体 |
| 5 | 4 | barren，裸地 |
| 6 | 5 | forest，森林 |
| 7 | 6 | agricultural，农业用地 |

规则由 `utils/constants.py` 集中管理：`map_raw_mask` 负责原始标签转训练索引，`encode_loveda_mask` 负责预测 0～6 转回 1～7。模型固定输出七类；**训练类别 0 是有效的背景类，不能当成 ignore。**

## 4. 为什么默认改为原分辨率随机裁块

1024→256 整图缩放会使每个方向缩小四倍。窄道路、小建筑与边缘可能在缩放后只剩少数像素甚至消失；把 logits 放大回去无法重新找回输入中丢失的细节。这是本项目选择裁块训练的技术判断，并非已经做过对比实验的结论。

默认 `--data-strategy crop --image-size 256 256`：在原始分辨率上随机取 256×256 窗口，保留像素尺度，并以较小块控制显存。每张原图每轮取样 4 次，即本地训练集每轮 10088 个样本；这些窗口可能重叠，不保证一轮覆盖整张原图。文件不会被裁剪后另存，也不会复制原始数据。

取舍是单块上下文范围有限。显存允许时可尝试 384 或 512，并相应减小 batch size；高宽须至少为 32 且均为 16 的倍数。较大的 crop 是否提高效果需实际实验，不能仅凭结构断言。

保留 `--data-strategy resize` 对照模式。该模式的 Val image 和 mask 同步缩小，指标在缩小网格上计算；**不能把它与默认 crop 模式在原图网格上的 mIoU 直接作为同口径排名。** 默认 crop 的 Val/Test 保持完整分辨率，通过滑窗避免整图网络前向带来的大特征显存。

## 5. 图像和 mask 的处理全过程

图像路径：

```text
磁盘 PNG → PIL 读取 → convert("RGB")
→ 原生随机 crop（或对照模式 bilinear resize）
→ Train 同步几何增强 → 可选仅图像的颜色增强
→ NumPy float32 / 255 → Tensor [3,H,W]
→ ImageNet normalization → DataLoader
```

归一化 mean=`[0.485,0.456,0.406]`，std=`[0.229,0.224,0.225]`。共享 `preprocess_image` 只在明确传入尺寸时 resize；Dataset 已完成几何变换后不重复缩放。使用 ImageNet 归一化不等于加载 ImageNet 预训练权重，本模型从随机初始化开始。

标签路径：

```text
mask PNG → 保留二维整数类别索引
→ 检查原图宽高和 0～7 范围
→ 使用与 image 完全相同的 crop / flip / rotation
→ 只有 resize 模式才使用 nearest interpolation
→ map_raw_mask：0→255、1～7→0～6
→ torch.long [H,W] → DataLoader
```

不把 P 模式调色板 mask 转为灰度颜色值，防止类别索引改变。裁剪与直角旋转本身不需要插值；mask 从不使用 bilinear。

小于 crop 的输入在右侧/底部补齐：图像复制边缘，原始 mask 填 0，随后统一映射为 ignore。随机 crop 如果全为 ignore，最多尝试 10 个窗口；最终仍为空则由训练循环处理，不无限重试，不假装实现类别均衡采样。

## 6. 适度的数据增强与 DataLoader

Train 默认启用同步水平翻转（概率 0.5）、垂直翻转（概率 0.5），方形块随机旋转 0/90/180/270 度。矩形块只选 0/180 度，保证 batch 高宽一致。PIL 图像与 NumPy mask 使用相同随机选择。`--no-augment` 关闭这些增强，但 crop 模式仍会随机选取裁剪位置。

`--color-jitter` 默认为 0，可设置小幅度（如 0.1）；只改变图像亮度、对比度、饱和度，mask 不变。它受 `--augment` 总开关控制。Val/Test 无随机增强；默认保持全图，Dataset 返回规则如下：

- Train/Val：`(float32 image, long mask)`。
- Test：`(float32 image, str(image_path))`，不伪造 mask。

Train loader 默认 batch size=2、shuffle=True；Val loader 固定 batch size=1、shuffle=False，以便保存完整且可能不同尺寸的原图。CUDA 时 pin_memory=True，送入 GPU 时 non_blocking=True。Windows 默认 num_workers=0；指定大于 0 时启用 persistent_workers，并在顶层 `seed_worker` 中为 Python/NumPy 随机增强播种。所有入口保留 `if __name__ == "__main__":`。

## 7. U-Net 3+ 架构与论文的对应关系

五层 Encoder 均为两次 `Conv3×3 → BN → ReLU`，相邻层之间为 2× MaxPool。

| 特征 | 通道 | 符号尺寸 | 256 输入示例 |
| --- | --- | --- | --- |
| E1 | 64 | H×W | 256×256 |
| E2 | 128 | H/2×W/2 | 128×128 |
| E3 | 256 | H/4×W/4 | 64×64 |
| E4 | 512 | H/8×W/8 | 32×32 |
| E5 | 1024 | H/16×W/16 | 16×16 |

每个 Decoder 节点融合五个尺度，保留[原论文第 2.1 节](https://arxiv.org/pdf/2004.08790)的核心连接关系：

| Decoder | 五路来源 | 目标尺寸 | 输出通道 |
| --- | --- | --- | --- |
| D4 | E1、E2、E3、E4、E5 | H/8×W/8 | 320 |
| D3 | E1、E2、E3、D4、E5 | H/4×W/4 | 320 |
| D2 | E1、E2、D3、D4、E5 | H/2×W/2 | 320 |
| D1 | E1、D2、D3、D4、E5 | H×W | 320 |

每条分支：`MaxPool 或 bilinear 对齐尺寸 → Conv3×3 + BN + ReLU → 64 通道`；五路 `torch.cat(dim=1)` 得到 320 通道，再经 `Conv3×3 + BN + ReLU` 融合。每条分支均在 forward 中显式书写，附符号 H/W 与 256 示例，没有用循环生成 Decoder。

## 8. 五级 Deep Supervision 与 CGM 取舍

[作者提供的 DeepSup 实现](https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/UNet_3Plus.py)包含最深层 hd5 的输出。因此新增 E5 分类头：D1 为 main，D2/D3/D4/E5 为四个 aux。E5 提供更深层语义监督，但其空间分辨率较低，默认给予较小权重，避免粗监督过度影响细节；是否有收益仍需消融实验。

```python
outputs = model(images)
# main: [B,7,H,W]
# aux: [d2_logits, d3_logits, d4_logits, d5_logits]
# 四个 aux 均插值到 [B,7,H,W] 后才与同一 GT 计算 loss。

outputs = model(images, return_aux=False)
# 验证/推理：返回相同 main 与空 aux，仅省略辅助分类头。
# Encoder 和四层 Decoder 仍会完整计算。
```

保留当前项目的 1×1 分类头，D1～D4 输入通道 320，E5 输入通道 1024；与论文中的 3×3 分类头不同。融合块已经使用空间卷积，此处采用直接通道分类，保持现有代码直观。这是明确的适配选择，不声称逐行复现论文。所有 head 返回 logits；没有 sigmoid 或 softmax 输出激活。

默认不加入 CGM。[原论文第 2.3 节](https://arxiv.org/pdf/2004.08790)的 CGM 判断图像是否含目标器官，以二分类结果抑制无器官图像的误分割。LoveDA 一张图可以同时包含多类地物，七类之间也没有同一套“器官/非器官”定义。直接硬门控可能将稀少道路或小建筑整类清零。可以另外研究七类多标签存在性辅助任务，但需独立监督、损失和消融，本版不引入该额外任务。

## 9. Loss：论文思想与 LoveDA 适配

论文 Hybrid Loss 结合 Focal、MS-SSIM 和 IoU，分别关注像素、局部结构与整体重叠。[原论文第 2.2 节](https://arxiv.org/pdf/2004.08790)

本项目默认 **CE + Dice**，并保留可选 Focal：

| 损失 | 本项目选择与原因 |
| --- | --- |
| CrossEntropy | 默认启用。适合每像素互斥七分类，直接接收 logits，支持 ignore 与类别权重 |
| Dice | 默认启用。对各类重叠等权求平均，补充像素 CE 对大面积区域的偏向 |
| Focal | 可选，默认权重 0。用于以后验证难像素/类别不均衡的影响，不默认叠加重复像素目标 |
| IoU loss | 不另加。与 Dice 的重叠目标相近，先保持一个清楚的重叠项；mIoU 仍是验证指标 |
| MS-SSIM | 不加入。类别编号没有连续灰度语义，不能直接在 0～6 编号图上算结构相似度；逐类 one-hot 改造还需正确处理 ignore、空类和多尺度窗口 |

每个 head 的 `CombinedLoss = ce_weight × CE + dice_weight × Dice + focal_weight × Focal`，默认权重为 1、1、0。模型输出 logits；CE 前不手动 softmax；Dice 内部使用 softmax；Focal 从未加权 log_softmax 取得 p_t，再施加类别权重。

训练总损失：

```text
L = L(D1) + 0.5 L(D2) + 0.25 L(D3) + 0.125 L(D4) + 0.0625 L(E5)
```

辅助权重是本项目的可配置选择，不宣称是论文固定权重。loss 内部用 FP32 计算 CE/softmax/累加，Dice epsilon=1e-6。ignore 像素从 CE/Focal、Dice 的概率与 one-hot 两侧同时排除。全 ignore 返回可微零值，训练循环进一步跳过参数更新。

Dice Loss 在 batch+空间维度汇总，各类等权；没有真实像素的类别仍保留其预测误报项。验证指标则把既无真实也无预测像素的类别排除宏平均，两者的空类规则不同。

## 10. 类别不均衡：使用真实 Train 频率生成权重

项目已有 `class_stats.json`，记录 Train 2522 张原始 mask 的真实统计，包含 Rural、Urban 和总计。
`compute_class_stats.py` 负责统计，不做 resize/crop、不写数据目录。本次读取已有 JSON，没有重新扫描或修改原始数据。

`train.py --class-weights` 支持三种用法：

- 不传参数：保持默认无类别权重，损失仍为 CE + Dice。
- `--class-weights auto`：读取 `train.py` 同目录的 `class_stats.json`，使用 `total.class_frequencies`。
- `--class-weights w0 w1 w2 w3 w4 w5 w6`：保持手动输入七个有限正数的方式，不额外归一化手动权重。

自动权重的类别顺序为 background、building、road、water、barren、forest、agricultural：

```text
frequency[c] = Train 中第 c 类像素数 / Train 有效像素总数
raw_weight[c] = 1 / sqrt(frequency[c])
class_weight[c] = raw_weight[c] / mean(raw_weight)
```

七个自动权重的均值约为 1；使用平方根倒数降低直接频率倒数的激进程度。
采用两域合计像素的频率，不对 Rural/Urban 频率做简单平均。原始 no-data=0（映射后 ignore=255）不计入频率或权重。
读取时检查 Train、类别顺序、ignore_index、七个正的有限频率及频率和；缺文件、零频率、null 或格式错误会明确报错。
启动时逐类打印名称、frequency 和 class weight；不修改 JSON 中记录的历史数据路径。

这些权重通过现有 `CombinedLoss` 同时传给 CE 和 Focal（若启用），并随损失模块移动到训练设备；Dice 保持各类等权。
默认仍是 CE + Dice，Focal 权重仍为 0。Focal 可通过 `--focal-weight` 开启，gamma 默认 2；若要用 Focal 替换 CE，可设 `--ce-weight 0 --focal-weight 1`。

以下命令供之后在已有、已确认的 WSL Python 环境中执行，本次未运行：

```bash
cd /home/kxcc/unet3
python train.py --device cuda:0 --class-weights auto --checkpoint-dir checkpoints/auto_weights
```

checkpoint 的 `config.class_weights` 保存解析后的七个数值，同时记录自动模式的 `class_frequencies`。
恢复时继续使用 `--class-weights auto` 并保持统计频率不变；原有 resume 校验会拒绝不一致的权重。
不使用类别权重的旧实验与启用 auto 的实验配置不同，应分别保存，不能直接当作相同配置续训。
不要使用 Val/Test 设计训练权重。

均匀随机裁块有助于保留细节，但不保证稀有类均衡；当前没有类感知采样、域平衡采样或 OHEM。是否需要这些策略，应根据统计、逐类验证指标和对照实验决定。

## 11. 一个样本从 PNG 到 Loss 的完整数据流

```text
同名 PNG image + PNG mask
  → LoveDADataset：按文件名配对、检查原始尺寸/标签
  → 同步随机 crop，必要时补边
  → 同步翻转 / 直角旋转，图像可选颜色增强
  → image: float32 /255 + normalize；mask: 0→255、1～7→0～6
  → DataLoader: [B,3,H,W] + [B,H,W] long
  → CUDA：model / image / mask / loss 权重位于同一设备
  → autocast：Encoder → 全尺度 Decoder → 五个 logits
  → 所有 logits 与 GT 尺寸一致
  → FP32 CE + Dice（可选 Focal）
  → main + 四个带权 aux = total loss
  → scaler.scale(loss).backward()
  → scaler.unscale_(optimizer) → 可选梯度裁剪
  → scaler.step(optimizer) → scaler.update()
```

## 12. CUDA、AMP、优化器与 Scheduler

训练默认 `--device cuda:0`，模型和损失对象显式 `.to(device)`；图像、mask 使用 `.to(device, non_blocking=True)`。请求的 CUDA 不可用时明确报错，不把默认 GPU 训练静默切到 CPU。用户可明确指定 `--device cpu`。结构脚本与推理默认 auto；训练也可显式传 auto，此时允许自动选择 CPU。

启动后打印 CUDA available、device、GPU name、CUDA device count、AMP enabled。此次没有启动这些入口，**没有验证本机 CUDA 是否可用**。

CUDA 默认启用 AMP；`--no-amp` 关闭。CPU 自动关闭 AMP。autocast 使用 FP16，模型参数仍保持常规 FP32；损失中的概率与归约转 FP32。GradScaler 优先用新版 `torch.amp.GradScaler`，旧 PyTorch 2.0/2.1 走 `torch.cuda.amp.GradScaler` 兼容分支。

梯度裁剪默认最大 L2 范数 1.0，可用 `--grad-clip 0` 关闭。先 unscale 再裁剪，避免把缩放后的梯度按错误阈值裁剪。AMP 溢出由 scaler 跳过更新并降低 scale，日志统计跳过批次。[PyTorch 官方 AMP 教程](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

保留 AdamW：学习率 3e-4、weight decay 1e-4，作为简单可调的起点。保留单一 CosineAnnealingLR：T_max=epochs，eta_min=0，每轮训练/验证后调用 step。它们不是复现论文医疗实验的参数，也不是已验证的 LoveDA 最优超参数。

## 13. 验证、指标和推理

默认验证对整张原图使用 256×256 tile、128×128 stride；末端窗口锚定边缘以保证全覆盖，小图补边后再裁回。每个 tile 单独送 GPU，完整特征不驻留 GPU；main logits 转 FP32 后在 CPU 累加并除以覆盖次数。

```text
完整 RGB → 归一化 → 确定性滑窗 → GPU / AMP / main logits
→ CPU 平均重叠 logits → 完整 [1,7,H,W]
→ 如有需要先 interpolate logits → 最后 argmax
→ [1,H,W] 类别图 → 与完整 GT 累计混淆矩阵
```

Val 使用 eval/no_grad，滑窗函数也关闭梯度；不会更新 BatchNorm。**重叠窗口不各自重复计分**，只对重建整图更新一次混淆矩阵。矩阵行为 GT、列为 prediction，int64 计数，float64 计算指标：

- IoU = TP / (TP+FP+FN)。
- Dice = 2TP / (2TP+FP+FN)。
- 255 不参与任何计数；无 GT 且无 prediction 的类显示 N/A 并排除均值。
- 仅有 prediction、没有 GT 的类计为 0；整套验证无有效标签时直接报错。

日志 `Train Loss` 含辅助项，按有效批次样本数加权平均；`Val Main Loss` 是整图 main 的 CombinedLoss，按有效图像平均。两者口径不同，不应直接比数值大小判断过拟合；mIoU/Dice 是整个验证集累计混淆矩阵的宏平均。

默认 1024 图像有 7×7=49 个窗口，本地 Val 共 81781 个窗口，因此比整图缩为 256 的旧验证更耗时。可在一次新实验中用 `--eval-stride 256 256` 降到 16 窗口/图，但重叠减少，边界结果可能变化。上述仅是按尺寸计算的窗口数，没有做速度 benchmark。

`predict_image(image_path, checkpoint_path, device=None, restore_original_size=True, amp_enabled=None)` 返回 uint8 `[H,W]`。它读取 checkpoint 的策略、尺寸、stride、类别顺序与归一化配置；crop 使用整图滑窗，resize 使用与训练相同的 PIL bilinear 预处理。默认沿用 checkpoint AMP 设置，CPU 始终关闭；CLI `--no-amp` 可强制关闭。

默认保存到 `predictions/<stem>_prediction.png`，标签为 0～6；`--loveda-labels` 改为 1～7。Test 没有 no-data 真值，不能恢复真实 0 区域。禁止覆盖已有输出、输入图片或写入 dataset。`--keep-training-size` 只适用于 resize checkpoint；crop 模式需输出完整原图，使用该参数会明确报错。

## 14. Checkpoint 与 resume

每轮保存 `last_checkpoint.pth`；验证 main mIoU 严格提高时保存 `best_model.pth`。两者均为完整字典：

```text
model_version, epoch（已完成轮数）, model_state_dict,
optimizer_state_dict, scheduler_state_dict, scaler_state_dict,
best_miou, val_loss, val_miou, val_dice, config, rng_state
```

AMP 开启时保存 scaler 状态，关闭时保存 None。配置包括数据策略、尺寸、stride、增强、损失、类别顺序、归一化和训练参数。保存先写临时文件再替换，降低写入中断损坏已有 checkpoint 的风险。加载使用 `weights_only=True`，状态只由 Tensor 和基础 Python 类型构成。

resume 恢复模型、optimizer、scheduler、可用 scaler、主进程 Python/NumPy/torch/CUDA 和训练 DataLoader generator 状态。恢复参数需与保存时一致，尤其 epochs、batch、crop、stride、增强和 loss；设备/worker 数可调整，但不承诺逐位重现。原训练关闭 AMP、本次启用时创建新 scaler；反向切换到 CPU/FP32 时不使用 scaler。恢复时还要求同一实验目录保留 `metrics.csv`，否则无法可靠重建最佳 epoch、历史曲线与 summary。

默认 num_workers=0 时主进程随机状态可接续；persistent workers 内部随机流未保存，重建 worker 也可能改变后续取样顺序。CUDA 随机状态按 GPU 编号恢复，切换 GPU 编号不保证接续同一随机流。不同 worker 数、硬件、CUDA 算法和 AMP 选项可能产生不同随机序列或数值结果。本项目提供 epoch 边界续训，不提供 batch 中途精确恢复。

模型版本为 `unet3plus_loveda_v2`。旧四输出 checkpoint 缺少 E5 分类头，不能直接当成本版本 resume 或静默忽略参数；当前入口明确拒绝旧版本。此次检查 checkpoints 中只有 `.gitkeep`，没有已有训练权重被改动。

## 15. 实验结果记录与可视化

所有实验结果统一写入 `--checkpoint-dir`：

```text
checkpoints/实验名/
├─ best_model.pth
├─ last_checkpoint.pth
├─ metrics.csv
├─ config.json
├─ summary.json
├─ confusion_matrix.csv
├─ confusion_matrix.png
├─ curves/
│  ├─ loss_curve.png
│  ├─ miou_curve.png
│  ├─ dice_curve.png
│  ├─ lr_curve.png
│  ├─ per_class_iou_curve.png
│  └─ per_class_dice_curve.png
└─ predictions/
   ├─ best/
   └─ epoch_0005/, epoch_0010/, ...
```

`config.json` 在训练开始时保存全部命令行参数和解析后的配置，包括 epochs、batch size、crop size、数据策略、验证 stride、学习率、类别权重、各损失权重、辅助头权重、AMP、seed、固定验证样本等。`metrics.csv` 每轮验证结束后立即追加：epoch、Train Loss、Val Main Loss、mean Dice、mIoU、当轮学习率，以及七类 IoU/Dice。

根目录 `confusion_matrix.csv` 保存最近完成一轮 Val 的原始像素计数，行是真实类别、列是预测类别；PNG 为按真实类别行归一化的热力图。`summary.json` 每轮更新，完整训练结束时状态改为 `completed`；`best_epoch` 按 mIoU 选择，`best_dice` 取历史最高 Dice，`best_val_loss` 取历史最低验证损失，并分别记录对应 epoch。它还保存 best-mIoU epoch、最终以及逐类历史最高 IoU。

默认固定选择 6 张 Val 图像：Rural 3 张、Urban 3 张，各域按已排序文件列表等间隔选择，不依据预测效果。验证循环直接复用本轮整图滑窗产生的预测，不额外做模型 forward。每 5 个 epoch（可用 `--visualization-interval` 修改）以及最后一轮保存 RGB、固定颜色 GT、固定颜色预测和带类别图例的三联图；每次 best mIoU 更新时同步覆盖 `predictions/best/`。`--visualization-samples` 可设为 5～10。

训练完整结束后才从 `metrics.csv` 生成六张曲线，避免每个 epoch 重复绘图。每轮只新增一次 CSV 行、一次混淆矩阵图，以及按间隔或 best 条件保存少量验证图；相对完整滑窗验证，额外计算量很小，但 PNG 编码会带来少量 CPU 和磁盘开销。

## 16. 默认参数与之后的手动命令

| 参数 | 默认值 |
| --- | --- |
| 训练 device / AMP | cuda:0 / CUDA 开启 |
| epochs / batch size | 50 / 2 |
| 数据策略 / crop | 原生随机 crop / 256×256 |
| samples_per_image | 4 |
| Val batch / stride | 1 / 128×128 |
| 几何增强 / color_jitter | 开启 / 0 |
| num_workers / persistent_workers | 0 / workers>0 时开启 |
| AdamW lr / weight_decay | 3e-4 / 1e-4 |
| Cosine T_max / eta_min | epochs / 0 |
| CE / Dice / Focal 权重 | 1 / 1 / 0 |
| class_weights / focal_gamma | None（可选 auto 或七个手动权重）/ 2 |
| aux D2/D3/D4/E5 | 0.5 / 0.25 / 0.125 / 0.0625 |
| Dice epsilon / grad_clip / seed | 1e-6 / 1.0 / 42 |
| 固定 Val 样本 / 保存间隔 | 6 / 每 5 epoch |

以下均在项目根目录执行，**本次未执行**。

```text
# 推荐起点：原生 256 裁块 + CUDA + AMP，输出到新的实验目录
python train.py --device cuda:0 --checkpoint-dir checkpoints/crop256_new

# 恢复上述实验；若首次改过参数，此处需传相同训练参数
python train.py --device cuda:0 --checkpoint-dir checkpoints/crop256_new --resume checkpoints/crop256_new/last_checkpoint.pth

# 显存允许时再尝试更大视野；不要假设任意 GPU 都能容纳
python train.py --device cuda:0 --image-size 512 512 --batch-size 1 --checkpoint-dir checkpoints/crop512

# 整图缩放对照；注意其验证网格与 crop 模式不同
python train.py --device cuda:0 --data-strategy resize --samples-per-image 1 --checkpoint-dir checkpoints/resize

# CUDA FP32 或明确指定 CPU
python train.py --device cuda:0 --no-amp --checkpoint-dir checkpoints/fp32_new
python train.py --device cpu --batch-size 1 --checkpoint-dir checkpoints/cpu_new

# 手动结构检查：五个输出应为 [2,7,256,256]
python test_model.py --device auto
python test_model.py --device cuda:0 --image-size 256 512 --batch-size 1

# 将图片路径替换为真实文件名
python predict.py dataset/Test/Rural/images_png/你的图片.png --device cuda:0
python predict.py dataset/Test/Urban/images_png/你的图片.png --device cuda:0 --loveda-labels
```

新实验必须使用新的或空的 checkpoint-dir，已有内容时入口会拒绝覆盖。改变总 epochs 不是当前 resume 的支持方式，因为会改变原来的余弦调度周期；若要从权重开启新的微调实验，使用第 20 节的 `--init-checkpoint`。

依赖为 torch、Pillow、NumPy、tqdm 和 matplotlib；matplotlib 只用于生成实验曲线、混淆矩阵与验证对比图，增强仍由 PIL/NumPy 实现，无需 torchvision。当前项目已迁移到 WSL 的 `/home/kxcc/unet3`；Linux 命令应使用用户已有、已确认的 WSL Python 环境。迁移前的 Windows 解释器路径不再作为当前运行要求。本次没有安装、升级、降级依赖或重建环境，也没有执行 Python 或验证 GPU 运行情况。

## 17. 静态审阅覆盖与尚未验证的内容

已通过阅读源码核对模块导入、五输出与四辅助权重、long 标签和 ignore 常量、几何同步、无标签 Test、GPU/AMP 调用顺序、scaler 保存恢复、main-only 验证、滑窗覆盖及先 logits 后 argmax。这里的“静态审阅”仅指阅读，不代表 lint、测试或运行通过。

仍需用户以后实际验证：依赖/驱动是否兼容、CUDA 是否可用、输入文件是否全部完整、实际显存峰值、完整前向与反传、AMP 数值表现、训练速度、checkpoint 实际恢复、滑窗边界效果、逐类指标和收敛质量。随机初始化、裁块视野、类别不均衡、BatchNorm 小 batch 统计、E5 辅助权重均可能影响最终效果，需通过真实实验判断。

## 18. 展示图片预留与运行交接

2026-09-17 新增 `demo_assets/Rural`、`demo_assets/Urban`，各存放 20 张 Test 原图副本；共 40 张，没有移动或修改原始数据。选取方法为各域按文件名排序后均匀间隔取样，不依据模型预测效果筛选。`demo_assets/manifest.json` 记录来源、唯一展示 ID、尺寸及 SHA-256；副本已与源文件逐一核对一致。

这些图片没有真实标签，只用于训练完成后的展示推理，不用于训练、验证、类别权重或调参。页面尚未开发，预测尚未生成。具体说明见 `demo_assets/README.md`。

`ANTIGRAVITY_RUN_PROMPT.md` 提供后续监督执行的完整提示词，要求先确认项目解释器与 CUDA，再做有明确更新步数、验证图像数上限的短检查，检查通过后进入正式训练。当前 `train.py` 尚无 smoke/max-steps 参数，不能把完整 `--epochs 1` 当作短检查。此次只准备展示副本和交接文档，没有实际执行项目代码或安装依赖。

## 19. 用已有最佳权重做小规模诊断（WSL）

`diagnose.py` 用于决定下一轮实验方向。它读取已经训练好的 checkpoint，不创建 optimizer，不反向传播，不修改网络、Dataset、Loss、深监督或训练调度器，也不覆盖原始权重。默认从 Val 的 Rural、Urban 各固定随机抽取 50 张，共 100 张，以同一批图片比较四组设置：

| mode | BN 统计 | 滑窗步长 |
| --- | --- | --- |
| baseline | checkpoint 原始统计 | checkpoint 中的 eval_stride |
| overlap | checkpoint 原始统计 | 默认 crop 高宽各一半 |
| bn | 仅用 Train 重新校准 | checkpoint 中的 eval_stride |
| bn_overlap | 与 bn 共用校准后的统计 | 默认 crop 高宽各一半 |

对于 `checkpoints/auto_weights_40e/best_model.pth` 的配置，四组均保持 256×256 裁块，比较 256×256 与 128×128 步长。重叠区域平均原始 logits，融合结束后才 argmax。重叠可能缓解裁块边缘接缝，但会增加窗口数：1024×1024 图像从 16 个窗口变成 49 个窗口，不能保证精度一定提升。

BN 校准是一个待验证假设：小 batch 训练下的运行统计可能影响验证表现。脚本重置 BN 的 running mean/variance，以 `momentum=None` 累计 128 个 batch，每批 2 个原生随机 Train 裁块；Rural/Urban 各选 128 张原图并混合顺序，不启用翻转、旋转或颜色增强。仅 BN 临时进入 train 模式，其余模块保持 eval，卷积与 BN affine 参数均不更新。校准不使用 Val/Test 图片或 Val 标签；也不保证优于原始统计。校准统计只在本次进程中保留，不生成可误用为 resume 的 checkpoint。

以下命令由用户在 **WSL Ubuntu 终端**中手动执行，明确使用现有 WSL 环境 `/home/kxcc/venvs/unet3/bin/python`。无需另建环境或重新安装 torch：

```bash
cd /home/kxcc/unet3
/home/kxcc/venvs/unet3/bin/python diagnose.py \
  --checkpoint checkpoints/auto_weights_40e/best_model.pth \
  --device cuda:0 \
  --val-samples 100 \
  --tile-batch-size 2 \
  --bn-batches 128 \
  --bn-batch-size 2 \
  --num-workers 2 \
  --seed 42
```

AMP 默认沿用 checkpoint，CPU 自动关闭。`--tile-batch-size 2` 把两个滑窗合并前向；默认在计算设备上累加 FP32 logits，最后一次性传回 CPU。现有 `train.py` / `predict.py` 未传入这些新选项，仍使用原来的逐窗口、CPU 融合行为。扩大窗口批量可能产生微小浮点差异；四组对照使用相同批量设置。若显存不足，将命令中的窗口批量改为 `--tile-batch-size 1`；必要时加 `--no-accumulate-on-device`。严格复用原始滑窗执行方式时，同时使用这两个选项。

每次默认新建 `checkpoints/diagnostics_日期_时间/`，开始评估前打印实际路径。也可以用 `--output-dir` 指定项目内一个尚不存在的目录；拒绝覆盖已有目录或写入 dataset。

```text
checkpoints/diagnostics_日期_时间/
├─ config.json          # 命令行参数、来源 checkpoint 配置、实际 stride/AMP
├─ samples.json         # 固定 Val 名单、Train 校准名单与随机种子
├─ comparison.csv       # 每组 Overall/Rural/Urban 的 Loss、mIoU、Dice、逐类指标
├─ report.json          # 已完成组的结果、checkpoint epoch、分阶段耗时
├─ baseline/
│  ├─ Overall/confusion_matrix.csv、confusion_matrix.png
│  ├─ Rural/confusion_matrix.csv、confusion_matrix.png
│  ├─ Urban/confusion_matrix.csv、confusion_matrix.png
│  └─ predictions/      # 相同 6 张图的 RGB、GT、预测、带图例三联图
├─ overlap/             # 同上
├─ bn/                  # 同上
└─ bn_overlap/          # 同上
```

`comparison.csv` 和 `report.json` 每完成一组就保存，已完成的对照不会因之后手动停止而丢失。耗时分为 checkpoint 加载、Train BN 校准、各组评估和各组绘图写盘；评估时间包含取图、滑窗、损失、指标和少量可视化数据整理。CUDA 在阶段边界同步计时，但第一组还有首次算子初始化、后续组可能受文件缓存影响，所以这些数值用于估算开销，不是严格速度 benchmark。

先比较**同一个诊断目录**下的四组结果及接缝、barren/agricultural 等类别表现。100 张的均衡子集与完整 Val 的域比例不同，子集 mIoU 不能直接拿来和已有完整 Val 的 34.57% 比较。只有子集表现改善的方案才值得进一步做完整 Val 确认；不要依据子集分数声称正式精度提高。

例如，若 `bn_overlap` 在同批对照中更好，再执行下面命令验证完整 Val；它会按相同 seed、worker 数、BN 批数和批量重建同一套校准裁块，然后只评估这一组。若较好的是 `overlap`，把 `--modes bn_overlap` 改成 `--modes overlap`，即可跳过 BN 校准：

```bash
cd /home/kxcc/unet3
/home/kxcc/venvs/unet3/bin/python diagnose.py \
  --checkpoint checkpoints/auto_weights_40e/best_model.pth \
  --modes bn_overlap \
  --val-samples 0 \
  --device cuda:0 \
  --tile-batch-size 2 \
  --bn-batches 128 --bn-batch-size 2 \
  --num-workers 2 --seed 42
```

完整 Val 命令仍有较大计算量，请先看小规模诊断结果。再次运行默认生成新目录；BN 裁块复现还依赖文件列表、worker 数与软件环境一致。上述命令仅作为后续运行说明，本次修改只进行了源码静态审阅，没有执行诊断、forward、测试或训练。

## 20. 诊断结果与独立微调入口

项目现有 `checkpoints/diagnostics_20260919_142359_262161/report.json` 已记录完成的诊断。以下是读取现有结果得到的值，不是本次修改时重新运行的结果：

| 设置 | 100 张子集 mIoU | mean Dice | 评估耗时 |
| --- | --- | --- | --- |
| baseline | 32.1685% | 48.0623% | 53.15 秒 |
| overlap | 32.4551% | 48.3919% | 153.80 秒 |
| bn | 32.0242% | 48.0516% | 51.38 秒 |
| bn_overlap | 32.3087% | 48.3718% | 154.03 秒 |

重叠滑窗的增益约为 0.29 个百分点，评估耗时约为同批 baseline 的 2.89 倍。当前 BN 校准使总体 mIoU 略降，农业类 IoU 从 34.18% 降到 27.57%；因此新微调命令不采用校准后的 BN 统计，也不把重叠滑窗当主要提分手段。这只说明此次校准设置没有改善总体结果，不能据此判断所有 BN 策略均无效。

原始子集的 Rural mIoU 23.85%，Urban 36.73%。这提示要关注乡村场景与类别混淆，但不能直接归因为 Rural 图片数量不足或证明采样不足。第一轮先尝试增加原生裁块覆盖、延长最佳权重附近的训练；384/512 更大视野、难类采样或其他结构变化留作后续独立实验。

`train.py --init-checkpoint PATH` 与 `--resume PATH` 的区别：

- `--init-checkpoint`：严格加载全部模型参数及原始 BN 统计，检查模型版本、七类顺序、ignore 与归一化；重新创建 AdamW、cosine scheduler、GradScaler，微调轮数从 1 开始，必须写入新目录。旧实验记录和权重不变。
- `--resume`：恢复同一实验的模型、optimizer、scheduler、scaler、随机状态和已完成轮数；配置一致性要求保留。resume 与 init-checkpoint 互斥，checkpoint-dir 必须与恢复文件所在目录一致。

微调时默认先运行一次**完整 Val**，保存 `initial_validation.json`，记录本次评估协议下的起点损失、mIoU、Dice、逐类 IoU。它不更新参数或 BN，不写成训练 CSV 的 epoch 0。之后每轮仍完整验证，以完整 Val mIoU 保存微调阶段最好的模型。

`summary.json` 额外保存 `initial_validation_miou`、`best_miou_gain_over_initial` 和 `improved_over_initial`。差值以 0～1 表示，乘以 100 才是百分点。新实验内部的最佳分数可能低于初始模型；只有正差值才能说明该完整 Val 协议下有提升，不能把“保存了 best_model”当作超过旧模型的证据。

新增 `timings.json` 每轮记录 Train、Val、保存/绘图耗时和 CUDA 峰值已分配显存 GiB；初始 Val 单独计时。计时包含对应阶段的数据读取或写盘，没有改变 optimizer.step / scheduler.step 的顺序，也没有降低验证分辨率或缩减正式验证集。最终生成六张曲线的时间不包含在各 epoch 的 logging_seconds 中。

建议先运行以下一组独立微调，作为待验证方案：

```bash
cd /home/kxcc/unet3
/home/kxcc/venvs/unet3/bin/python train.py \
  --init-checkpoint checkpoints/auto_weights_40e/best_model.pth \
  --checkpoint-dir checkpoints/finetune_best30_12e \
  --device cuda:0 --amp \
  --epochs 12 --batch-size 2 \
  --data-strategy crop --crop-size 256 256 \
  --samples-per-image 2 \
  --lr 5e-5 --weight-decay 1e-4 \
  --class-weights auto \
  --eval-stride 256 256 \
  --eval-tile-batch-size 2 --eval-accumulate-on-device \
  --num-workers 4 --seed 42
```

保留 CE=1、Dice=1、Focal=0，辅助头权重 `[0.5, 0.25, 0.125, 0.0625]` 和全部几何增强。lr=5e-5 与原实验最佳第 30 轮的约 5.26e-5 接近；这次 cosine 的 T_max 为新实验 12 轮。每图每轮从原实验的 1 次 crop 增至 2 次，12 轮累计新增每图 24 次取样，相当于原 40 轮取样总量的 60%，仍需要真实训练时间，不能承诺很快完成或一定提高多少分。

`--eval-tile-batch-size` 与训练 batch 独立；`--eval-accumulate-on-device` 复用诊断中的 FP32 设备融合。旧默认仍是窗口 batch=1、CPU 融合；新推荐命令显式启用 batch=2、设备融合。若仅验证阶段显存不足，可以改为窗口 batch=1，并关闭设备融合；此类执行设置可能产生微小浮点差异，应记录在配置中。不要把诊断的耗时比当成整个训练的固定加速比。

若上述微调中断，保持其它参数不变，用下面命令恢复同一实验，**不要再次传 init-checkpoint，也不要改 epochs**：

```bash
cd /home/kxcc/unet3
/home/kxcc/venvs/unet3/bin/python train.py \
  --resume checkpoints/finetune_best30_12e/last_checkpoint.pth \
  --checkpoint-dir checkpoints/finetune_best30_12e \
  --device cuda:0 --amp \
  --epochs 12 --batch-size 2 \
  --data-strategy crop --crop-size 256 256 \
  --samples-per-image 2 \
  --lr 5e-5 --weight-decay 1e-4 \
  --class-weights auto \
  --eval-stride 256 256 \
  --eval-tile-batch-size 2 --eval-accumulate-on-device \
  --num-workers 4 --seed 42
```

resume 需要保留新实验的 `metrics.csv` 和 `initial_validation.json`；不再重复初始验证，来源权重信息会从 checkpoint 配置恢复。若还没完成任何训练 epoch 就中断，尚无 last_checkpoint，可选择另一个新目录重新启动。

本次仅修改了入口、权重加载、结果记录与文档，没有改动模型、Dataset、Loss、深监督、AdamW 或 cosine 算法，也没有执行新增微调代码、训练、测试或安装依赖。运行后应联合比较完整 Val 的初始/最佳指标、逐类变化和耗时，不能只看 Train Loss 降低。
