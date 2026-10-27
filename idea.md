# WHU-MARS / UAD：SFTC-V2.1 最终优化版代码修改与实验实施指导

方案名称： SFTC-V2.1（Spatial–Global Feature Token Co-encoding，空间—全局特征共编码）

基础模型： WHU-MARS 官方 UAD，ImageNet 预训练 ViT-B/16

实验任务： WHU-MARS-1000 多模态多平台统一行人重识别（AS-ReID）

实施目标： 在不改变 UAD 原有身份监督和最终检索特征定义的前提下，引入轻量局部 CNN 提取空间细节，并通过一次来源校准注意力，使其参与共享 ViT 的身份表征学习。

本指导是在前面多轮方案审查、论文阅读、开源代码核对及 WQVD、TAUR 负结果反思基础上形成的首轮正式实验冻结版本。

## 一、实施前最后一轮审查

### 1.1 最终审查结论

从研究问题、网络机制、训练优化和推理契约四个角度重新检查后，我认为：

当前方案没有必须继续修改主体结构或增加辅助损失的明显缺陷，可以开始正式实施。

但必须把以下四件事区分清楚。

| 设计目标                 | 当前方案是否满足 | 需要正确理解的边界                |
| -------------------- | -------- | ------------------------ |
| 提供不同于原 ViT 的局部信息提取机制 | 是        | CNN 具有不同归纳偏置，但不保证提取到身份信息 |
| 让 CNN 有机会通过身份损失学习    | 是        | 梯度可达不代表实际梯度充分            |
| 降低来源选择过早排斥 CNN 的风险   | 部分满足     | 固定来源偏置并不能阻止 Q/K 调整注意力    |
| 保证新增信息提高最终检索效果       | 不能保证     | 必须由完整 WHU 检索实验验证         |

之前的 WQVD 和 TAUR 已经说明两种不同的失败可能：

- WQVD 中，局部融合可以对同一 Checkpoint 的 CLS 产生增益，但完整训练后的最终模型仍可能弱于独立 UAD。
- TAUR 中，中层模块可以正常训练，但最终开关对 CLS 的直接影响几乎可以忽略。

因此，这次不能再将“新模块可运行、梯度非零、Attention 有响应”当作方法成功的证据。

真正要检验的是：

\\[ \boxed{ \text{有效局部特征学习} \rightarrow \text{最终 CLS 利用这些特征} \rightarrow \text{身份检索性能改善} } \\]

### 1.2 最终确定的三个关键优化决策

决策一：采用来源偏置启动策略。

- Epoch 1–5：来源偏置固定为 −2.0。
- Epoch 6–120：来源偏置允许从 UAD 身份损失中正常学习。
- CNN、局部 K/V、ViT 从第一个 Epoch 开始共同训练。

它只能暂缓通过来源偏置主动关闭 CNN，不能保证局部注意力不降低，因为 Q/K 匹配仍然可以改变实际来源权重。

决策二：保留原 UAD 全局梯度裁剪。

官方代码使用 `clip_grad_norm_(model.parameters(), max_norm=1.0)`，位于 AMP 梯度反缩放之后。首轮保持这一规则，记录真实裁剪情况，不增加独立 CNN 裁剪，也不提前删除全局裁剪。

决策三：不增加新损失或新训练阶段。

保留 CE、WRT、ProCA、GPD，不增加 Local CE、额外 Triplet、蒸馏、正交约束或来源均衡损失。

这是为了让完整方法集中验证局部特征来源及其融合能力，而不是因为原四项损失已经保证新分支会学好。

### 1.3 本次明确不实施的旧设计

| 原方案或候选方法                        | 不作为首轮方案的原因                      |
| ------------------------------- | ------------------------------- |
| IDB：L4→L8 跨层注意力                 | 可能主要重复变换同一 ViT 已有信息             |
| Patch 与 CLS 双残差注入               | 存在重复注入及路径贡献不清的问题                |
| 新增局部 Token 长期留在 Block 8–12      | 后续所有层都接触新增 Token，初始来源控制不能覆盖整个网络 |
| 最终 CLS 与 Local Vector 相加或拼接     | 容易重新改变最终身份距离空间                  |
| 全模态多套大型 CNN                     | 参数和优化成本较大，当前证据不足                |
| 独立 Local CE / QCT               | 新增优化目标可能与统一身份表征产生张力             |
| MaskSub / M-SpecGene / 频域模块同时加入 | 会使首轮实验无法清楚检验当前方法的主要机制           |

所以本次只做一项完整结构改造：局部 CNN + Block 7 来源校准注意力。

# 二、参考论文及模块来源：必须明确区分“借鉴”与“自行设计”

SFTC-V2.1 并不是直接复现某一篇论文。其局部 CNN、注意力接口和启动训练策略都是针对 WHU-UAD 组合形成的方案。

下面列出每一项重要设计的理论依据、论文具体位置以及与原文的区别。

| 参考论文                                                                                                                                                                                            | 论文中的具体方法与位置                                    | SFTC 借鉴内容                          | 不直接照搬的部分                    |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------- | --------------------------- |
| [WHU-MARS / UAD](https://github.com/msm8976/WHU-MARS)（CVPR 2026）                                                                                                                                | 第4.2节 P×M×K 采样、第4.3节 ProCA、第4.4节 GPD、第4.5节总体损失 | ViT 基线、原始身份监督、原型记忆与统一检索协议          | 不改变 UAD 的四项损失               |
| [ViT-CoMer](https://openaccess.thecvf.com/content/CVPR2024/papers/Xia_ViT-CoMer_Vision_Transformer_with_Convolutional_Multi-scale_Feature_Interaction_for_Dense_CVPR_2024_paper.pdf)（CVPR 2024） | 第3.1节整体双分支结构、第3.2节 MRFP、第3.3节 CTI              | CNN 局部空间特征与 ViT 全局特征在主干内部交互的主要结构依据 | 不复制完整 MRFP、可变形注意力、双向多阶段 CTI |
| [FusionReID](https://arxiv.org/abs/2412.17239)（IEEE T-ITS 2025）                                                                                                                                 | DFE 双分支提取、DMF 双注意力融合，包含 LRU、HTM 及 MFU          | CNN–ViT 特征互补与跨来源注意力在 ReID 中的使用思路   | 不复制大型双主干、堆叠 HTM 和多个独立身份头    |
| [MSCMNet](https://www.sciencedirect.com/science/article/pii/S0031320324008410)（Pattern Recognition 2025）                                                                                        | 方法部分 MIMB、ALB，特别是 Fig. 2(b) 的浅深语义信息关联          | 局部、浅层信息可能未得到充分利用的研究动机              | 不复制四流 QFE、ALB 原公式或 QCT      |
| [SDS-Net](https://arxiv.org/abs/2506.06042)（IEEE TGRS 2025）                                                                                                                                     | 浅深层级信息协同、ADSF 自适应融合模块                          | 多层级空间信息提取与融合的参考                    | 不移植红外小目标检测头及其专用损失           |

ViT-CoMer 原论文的 MRFP 使用卷积多感受野特征，CTI 使用 CNN–Transformer 的交互；FusionReID 则采用 CNN/Transformer 特征提取及多次融合。它们能够支持采用互补编码结构的研究动机，但不能直接证明 SFTC 在 WHU 的 RGB/NIR/TIR 上有效。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com

+1



MSCMNet 主要关注可见光—红外 ReID 中浅层与深层语义信息的充分利用，本方案借鉴的是其问题意识，而不是其原始四流网络。

[image](https://www.google.com/s2/favicons?domain=https://www.sciencedirect.com\&sz=32)

ScienceDirect



### 开源代码的重点参考位置

- UAD： [CVPR26_UAD](https://github.com/msm8976/WHU-MARS/tree/main/CVPR26_UAD)，这是最终改动的直接基础。
- ViT-CoMer： [comer_modules.py](https://github.com/Traffic-X/ViT-CoMer/blob/main/detection/mmdet_custom/models/backbones/comer_modules.py)，重点参考 CNN、MRFP、CTI 的特征提取与交互机制。
- FusionReID： [fusion.py](https://github.com/924973292/FusionReID/blob/master/modeling/fusion_part/fusion.py)，重点参考 `Attention_mix`、`Mutual_Fsuion_Unit`、`Heterogenous_Transmission_Module`。
- MSCMNet： [model.py](https://github.com/Hua-XC/MSCMNet/blob/main/model.py)，参考多尺度语义相关性挖掘。
- SDS-Net： [官方仓库](https://github.com/PhysiLearn/SDS-Net)，仅作为空间层级融合参考。

特别需要说明：SFTC 自己设计的16×8局部 Token、Block 7 单次局部 K/V 接入、联合 Softmax 来源偏置、模态仿射及5 Epoch偏置固定策略，不是上述论文中某个原生模块的直接复制。

后续论文写作应如实表述为受到这些工作的启发并进行任务适配，而不是直接声称使用了 ViT-CoMer 的 CTI 或 FusionReID 的 HTM。

# 三、最终网络结构与完整计算流程

## 3.1 整体网络

输入 RGB / NIR / TIR 图像

B×3×256×128 + 真实传感器模态 ID

原 UAD ViT

Patch Embedding

Block 1–6

129×768

原 Q/K/V

新增轻量 CNN

局部空间特征提取

多尺度融合与投影

128×768

新增局部 K/V

Block 7：来源校准 Attention

129个主干 Query → 257个 Key/Value 候选

联合 Softmax + 可学习来源偏置

输出129×768

原 ViT Block 8–12

仅处理原129个主干 Token

最终768维 pre-BN CLS

原 UAD：CE + WRT + ProCA + GPD

全模型必须满足三个核心要求：

1. CNN 是信息来源，不是第二套身份网络。 不新增 CNN CLS，不新增独立检索向量。
2. 局部 CNN 只在 Block 7 提供额外 Key/Value。 后续 Block 8–12 不保留独立 CNN Token。
3. 训练和正式推理使用同一网络结构。 都启用 CNN 与来源注意力，最终都提取同一个768维 CLS。

这里的“训练与推理一致”指模型计算路径和特征定义一致，不是要求训练、推理使用相同的随机数据增强。训练增强和测试预处理仍遵循原 UAD。

## 3.2 模块一：轻量多尺度局部 CNN

主要参考：ViT-CoMer 第3.2节 MRFP 的卷积多尺度表示思想；FusionReID 的 DFE/LRU；MSCMNet 的多尺度有效信息利用动机。下面的具体 CNN 是 SFTC 自行设计的简化结构，不是原论文中的同名模块。

新增 CNN 采用两个层级的局部空间特征，再融合为16×8的局部 Token 网格。

所有下表尺寸采用 `[B,C,H,W]`。

| 阶段       | 具体运算                                                          | 输出              |
| -------- | ------------------------------------------------------------- | --------------- |
| Stem     | Conv3×3，3→32，stride2 → GN(8组) → 模态仿射 → GELU                   | `[B,32,128,64]` |
| Stage 1  | DWConv3×3(s2) → GN(8) → GELU → PWConv(32→64) → GN(8) → GELU   | `[B,64,64,32]`  |
| Stage 2  | DWConv3×3(s2) → GN(8) → GELU → PWConv(64→128) → GN(16) → GELU | `[B,128,32,16]` |
| 多尺度对齐    | Stage 1 降采样2次；Stage 2 降采样1次                                   | 各为16×8          |
| 特征融合     | Concat(64+128) → Conv1×1(192→256) → GN(32) → GELU             | `[B,256,16,8]`  |
| Token 投影 | Conv1×1(256→768) → Flatten → PE → LN                          | `[B,128,768]`   |

具体实现约束：

- 所有新增卷积 `bias=False`。
- DWConv 的 `groups=in_channels`；3×3 步长卷积使用 `padding=1`。
- Stem 的 GN 设置 `affine=False`，其余 GN 使用 `affine=True`。
- 所有指定位置使用 GELU，不额外替换为 ReLU、SiLU 或其他激活。
- 不添加新的 BatchNorm2d。

### 多尺度特征对齐

使用固定低通核：

\\[ K=\frac{1}{16} \begin{bmatrix} 1&2&1\\\ 2&4&2\\\ 1&2&1 \end{bmatrix} \\]

对每个通道独立滤波，使用 `reflect padding=1`，再进行步长2采样。

Stage 1 的64×32特征处理两次，Stage 2 的32×16特征处理一次，最终统一为16×8。

这一设计的作用是降低后段尺度对齐过程中的混叠风险。

注意，Stem、Stage 1、Stage 2 仍使用步长卷积，因而不能宣称 CNN 整体具有无损或完整抗混叠特征提取能力。

### 模态条件化

仅在 Stem 的 GN 后加入：

\\[ F_m=\gamma_m\odot\operatorname{GN}(F)+\beta_m \\]

每个模态分别拥有32维 \\(\gamma_m,\beta_m\\)。

初始化为：

\\[ \gamma_m=1,\qquad\beta_m=0 \\]

这一初始化只表示模态仿射最初不额外改变 GN 输出，不代表整个卷积编码器是恒等映射。

后续 Stage 1、Stage 2 继续共享 CNN 参数，不增加 TIR 专属主干。

### 位置编码

对16×8网格生成固定二维正弦余弦位置编码：

\\[ E^{2D}\in\mathbb R^{128\times768} \\]

高度和宽度方向各使用384维。

最终局部 Token：

\\[ \boxed{ L=\operatorname{LN}(L\_{\mathrm{raw}}+0.1E^{2D}) } \\]

固定要求：

- 位置网格按照行优先顺序展开；
- 不增加局部 CLS；
- 位置编码不可学习；
- 位置系数固定0.1；
- 不使用空间硬对应 Mask。

CNN Token 具有显式位置表达，但不假设它与深层 ViT 的相同位置 Token 必然具有相同语义内容。

# 四、模块二：Block 7 来源校准注意力

参考：ViT-CoMer 第3.3节 CTI 的 CNN–Transformer 内部交互机制，以及 FusionReID 的跨来源注意力融合思想。SFTC 的“局部 K/V + 联合 Softmax + 模态来源偏置”是针对 UAD 的自行设计，并非 CTI 或 HTM 的原样复现。

这是整个实验最重要的代码修改位置。

## 4.1 只改变 Key/Value 候选，不更换原 Block

原 ViT 前六层输出：

\\[ T_6\in\mathbb R^{B\times129\times768} \\]

通过 Block 7 原有 QKV 线性层计算：

\\[ Q,K_D,V_D \in\mathbb R^{B\times12\times129\times64} \\]

局部 CNN 经过独立 K/V 投影：

\\[ K_L,V_L \in\mathbb R^{B\times12\times128\times64} \\]

然后计算：

\\[ S_D=\operatorname{scale}(QK_D^\top) \\]

\\[ S_L=\operatorname{scale}(QK_L^\top)+b\_{m,h} \\]

其中：

\\[ b\in\mathbb R^{3\times12} \\]

分别对应三种模态和12个注意力头。

最终：

\\[ \boxed{ P=\operatorname{Softmax}([S_D,S_L]) } \\]

Attention Logit 形状必须为：

\\[ [B,12,129,257] \\]

经过 Attention Dropout、Value 聚合、原输出 Projection 及原 Block 残差和 MLP 后，输出仍为：

\\[ [B,129,768] \\]

### 必须完整继承原 Attention 计算

开发时不能只保留 QKV，而遗漏：

- 原来的 `self.scale`；
- Attention Dropout；
- 输出 Projection；
- Projection Dropout；
- Block 的 `norm1` 和 `norm2`；
- DropPath；
- MLP 及残差顺序。

尤其应保留原 `attn.qkv`、`attn.proj` 的参数名称和权重加载关系，防止 ImageNet 预训练参数因模块重构而无法正确匹配。

## 4.2 为什么采用联合 Softmax？

联合 Softmax 让原 ViT 和 CNN 来源通过 Query–Key 匹配共同参与注意力竞争。

对单个 Query、Head，在 Dropout 之前：

\\[ A=(1-\rho_L)A_D+\rho_LA_L \\]

因此：

\\[ A-A_D=\rho_L(A_L-A_D) \\]

这表明新增 CNN 不只是提供额外信息，也会替换一定比例的原主干注意力响应。

这是设计选择，不是已经证明只有正收益的操作。

所以不能用“增加了局部信息来源”直接推断“最终身份信息一定更多、更好”。

## 4.3 不增加第二套门控

来源权重已经由：

\\[ \rho_L= \frac{e^{b\_{m,h}}Z_L} {Z_D+e^{b\_{m,h}}Z_L} \\]

决定。

它包含内容匹配强度与可学习模态先验。

第一版不再添加：

- 额外 MLP 来源门控；
- 固定来源比例；
- 强制来源均衡损失；
- CLS 专用第二条融合路径。

避免多个机制同时改变来源竞争，降低实现复杂度和优化目标冲突。

# 五、初始化和来源偏置启动策略

## 5.1 首次训练的正确初始化顺序

推荐按以下顺序实施：

1. 构造原 UAD 主干及 SFTC 模块。

   原 ViT-B/16 的 Patch Embedding、12个 Transformer Block、BNNeck 和分类器保持原结构。
2. 加载原 ImageNet ViT 预训练参数。

   检查原 `attn.qkv`、`attn.proj`、MLP、位置编码等关键参数是否实际加载成功，不能仅看到程序打印“pretrained loaded”就认为没有遗漏。
3. 初始化新增 CNN。

   卷积采用 Kaiming 初始化，最终256→768的1×1投影采用截断正态初始化，标准差0.02。GN、LN 按常规初始化。
4. 复制局部 K/V 初值。

   从已加载的 Block 7 `attn.qkv` 中取出 K、V 的权重和存在的偏置，分别复制给局部 K/V。

   此后局部 K/V 独立训练，不与原参数共享。
5. 初始化来源偏置。

   全部三模态、12个 Head 的初始偏置设为 −2.0。
6. 构造优化器和学习率调度器。

   必须确保所有新增参数已经注册，并正确加入优化器。

复制 K/V 是合理的首轮初始化方式，但不能声称已经证明它优于随机初始化。

而且，复制只发生在首次初始化时，恢复 SFTC Checkpoint 时绝不能再次覆盖局部 K/V。

## 5.2 冻结来源偏置五个 Epoch

Epoch 1–5

# −2.0

固定来源偏置

CNN、局部 KV、ViT 同时训练

Epoch 6–120

# 可学习

释放来源偏置

由最终 UAD 损失共同优化

这一措施仅暂缓模型通过偏置直接降低 CNN 来源权重。

由于 Q/K 仍在更新，局部来源份额仍可变化。不能称其为“保证 CNN 不被忽略”。

### 具体实现要求

来源偏置应从模型构造开始就注册为正式可训练参数，并进入优化器。

前5个 Epoch：

- 前向使用固定 −2.0；
- 反向计算后使其有效梯度为零；
- 来源偏置 Weight Decay 为0；
- 保证参数本身不漂移；
- CNN、KV、ViT 的梯度不受该操作阻断。

第6个 Epoch 解除偏置梯度限制即可。

不需要重建 DDP、优化器、调度器，也不重新初始化任何网络参数。

同时，不应根据 Epoch 5 的训练表现临时延长固定期。首轮按预先设定完成120 Epoch。

# 六、训练损失：不新增监督，保持原 UAD 契约

最终目标：

\\[ \boxed{ \mathcal L= \mathcal L\_{\mathrm{CE}} +\mathcal L\_{\mathrm{WRT}} +0.01\mathcal L\_{\mathrm{ProCA}} +\mathcal L\_{\mathrm{GPD}} } \\]

注意各项损失的输入位置：

| 损失    | 使用的特征        | 主要优化作用     |
| ----- | ------------ | ---------- |
| CE    | BNNeck 后分类输出 | 训练身份分类判别   |
| WRT   | pre-BN CLS   | 批内身份度量关系   |
| ProCA | pre-BN CLS   | 同身份跨模态中心聚合 |
| GPD   | pre-BN CLS   | 全训练身份原型判别  |

新增 CNN、局部 K/V 和来源偏置通过最终 CLS 接受这些损失的梯度。

## 6.1 为什么这次不增加 Local CE？

因为 CNN 不是独立用于身份检索的完整分支。

如果现在单独要求 CNN 完成身份分类，就引入了另一个优化目标，而这个目标可能鼓励其学习容易分类但跨模态不稳定的局部信息。

FusionReID 使用多路身份监督，但其模型结构和训练任务与当前 SFTC 不同，不能直接认为这些监督应该原样加入 UAD。

另一方面，不能据此断言任何辅助损失都没有价值。

首轮保留原 UAD 四项损失，是为了先验证新结构本身能否在统一身份目标下产生有效增量。

## 6.2 原型库不要调整

保持 ProCA、GPD 的原始实现与更新时机。

需要特别注意，官方代码使用的 `GPD_MOMENTUM=0.2` 与论文中旧原型系数的符号定义不完全相同：源码更新形式为旧原型系数0.8、新中心系数0.2。

不要因为论文里出现0.8，就直接将配置中的0.2改成0.8。

也不要给 CNN 建立第二套原型库。

# 七、优化器与梯度裁剪

## 7.1 优化器

保持 UAD 原 SGD、Momentum、120 Epoch、Warm-up + Cosine 训练安排。

| 参数                  | 首轮配置           |
| ------------------- | -------------- |
| 原 ViT 参数            | 沿用 UAD 原配置     |
| 新增 CNN              | 峰值基础学习率0.024   |
| 新增局部 K/V            | 峰值基础学习率0.024   |
| 来源偏置、模态仿射           | 峰值基础学习率0.024   |
| 新增 Conv/Linear 权重衰减 | 1e-4           |
| 新增 GN/LN、模态仿射、来源偏置  | Weight Decay=0 |
| SGD Momentum        | 原 UAD 设置       |
| 最大训练 Epoch          | 120            |
| Warm-up             | 5 Epoch        |

当前官方 UAD 的 `BASE_LR=0.008`，因此新增参数3倍峰值基础学习率为0.024。

但官方 Scheduler 的 Warm-up 初始学习率使用统一数值，因此实际训练中不能认为每一步都保持精确的3倍比例。

记录实际学习率即可，首轮不必重写调度器。

## 7.2 保留全局梯度裁剪1.0

根据官方 `processor.py`，训练采用：

反向传播 → AMP 反缩放 → 全局梯度裁剪 → 优化器更新。

首轮保留：

\\[ \operatorname{max\\\_norm}=1.0 \\]

对总梯度：

\\[ G=\sqrt{\sum_i\\|g_i\\|\_2^2} \\]

裁剪比例近似为：

\\[ s\_{\mathrm{clip}} = \min\left(1,\frac{1}{G+\epsilon}\right) \\]

所有参数梯度采用共同缩放。

这不是专门抑制 CNN，但可能降低 CNN/KV 的绝对更新幅度。

### 最终建议的训练操作顺序

1. 计算原 UAD 四项损失并反向传播。
2. AMP 反缩放梯度。
3. 若为 Epoch 1–5，将来源偏置梯度置零。
4. 记录裁剪前总梯度范数和必要的 CNN/KV 梯度统计。
5. 执行全模型梯度裁剪，阈值1.0。
6. 执行原优化器更新。
7. 按原 UAD 规则更新 GPD 原型库。

需要注意，来源偏置固定期间不能因为 Weight Decay 或优化器状态导致其数值变化。

### 为什么现在不修改裁剪阈值？

因为没有 SFTC 的实测证据证明1.0限制了有效训练。

当前若删除全局裁剪，同时改变来源偏置及新增参数学习率，会进一步扩大与原 UAD 的训练差异。

因此，保留该设置更合理。

不过必须记录：

- 裁剪前梯度范数；
- 裁剪系数；
- 触发裁剪的 Step 比例；
- CNN/KV 的实际参数更新情况。

不能用“CNN 梯度非零”或“新参数学习率3倍”来证明分支已获得充分优化。

# 八、完整代码文件修改指导

以下路径均相对于官方仓库中的 `CVPR26_UAD/`。

## 8.1 文件修改总览

| 文件                               | 修改类型   | 主要任务                         |
| -------------------------------- | ------ | ---------------------------- |
| `model/backbones/sftc_local.py`  | 新增     | 轻量 CNN、模态仿射、低通对齐、位置编码、局部 K/V |
| `model/backbones/vit_pytorch.py` | 修改     | 在 Block 7 接入局部 K/V 来源        |
| `model/make_model.py`            | 修改     | SFTC 构造、初始化、模态传参和输出契约        |
| `processor/processor.py`         | 修改     | 训练/推理模态 ID、偏置固定期、必要日志        |
| `solver/make_optimizer.py`       | 修改     | 新增参数组和学习率规则                  |
| `configs/UAD_SFTC_V2_1.yml`      | 新增     | 固定所有结构和训练配置                  |
| `datasets/*`                     | 原则上不修改 | 保留原 PKM 和数据顺序                |
| `loss/*`                         | 不修改    | 原 CE/WRT/ProCA/GPD           |
| `utils/metrics.py`               | 不修改    | 保留统一评测协议                     |

### 文件一：`sftc_local.py`

实现完整局部 CNN，包括：

- Stem、Stage 1、Stage 2；
- 三模态条件仿射；
- 两级特征的固定低通降采样；
- 16×8空间融合；
- 256→768投影；
- 固定二维位置编码；
- 局部 K/V 独立投影及初始化接口。

输出契约：

给定图像和模态 ID，提供供 Block 7 使用的128个768维局部 Token，或相应的局部 K/V。

模块不负责身份分类，不输出独立检索特征。

### 文件二：`vit_pytorch.py`

这是核心修改文件。

保留原 ViT 前六层和后五层，只在第7个 Block 增加局部来源读取。

特别注意 Python 的索引：

- `blocks[0:6]`：原 Block 1–6。
- `blocks[6]`：需要修改的 Block 7。
- `blocks[7:12]`：原 Block 8–12。

推荐在原 Attention 结构上加入可选的局部 K/V 输入，而不是替换为一套参数命名和计算顺序不同的新 Transformer。

必须支持明确的诊断关闭模式。

关闭时直接使用原主干 Q/K/V 计算 Attention，不加入局部候选。

### 文件三：`make_model.py`

负责正确完成：

- ViT 预训练权重加载；
- 新模块初始化；
- 局部 K/V 首次复制；
- 模态 ID 传递；
- 训练和推理输出保持原格式。

官方 `TransReID.forward_features` 的参数顺序为：

`x, camera_id, modal_id, view_id`

而对外 `TransReID.forward` 使用 `modal_label` 接口。

因此，不得将模态 ID 误传给 `camera_id`。

建议使用明确的命名参数，避免位置参数混淆。

同时，`mode` 只代表原模型训练/评估调用分支，不是模态编号。

训练时原返回仍为：

`cls_score, global_feat, feat`

其中 `global_feat` 始终为最终768维 pre-BN CLS。

### 文件四：`processor.py`

新增四类功能即可：

1. 构造训练时的模态 ID；
2. 验证与测试时传入真实 `modids`；
3. Epoch 1–5 固定来源偏置梯度，Epoch 6开始释放；
4. 增加少量有效学习状态日志。

其余原 UAD 训练流程保持不变。

特别注意，原训练图像按模态组织并拼接，ProCA 依赖这个顺序形成模态中心。

新增模态 ID 必须与真实拼接后的每张图像对应，不能只是生成一个长度正确但顺序错误的标签向量。

### 文件五：`make_optimizer.py`

识别新增参数，并配置其独立学习率及 Weight Decay。

需要保证：

- 原 ViT 参数仍沿用基线规则；
- 新增 CNN、KV 使用3倍峰值基础学习率；
- 来源偏置与模态仿射不衰减；
- 新增 bias 不重复乘原 `BIAS_LR_FACTOR`；
- 来源偏置从模型构造开始就在优化器内，Epoch 6不重新创建参数组。

### 文件六：独立配置文件

建立：

`configs/UAD_SFTC_V2_1.yml`

建议把以下内容固定到配置中：

| 配置项                             | 固定值       |
| ------------------------------- | --------- |
| `SFTC.ENABLED`                  | True      |
| `SFTC.BLOCK_INDEX`              | 6         |
| `SFTC.LOCAL_GRID`               | 16×8      |
| `SFTC.CNN_CHANNELS`             | 32、64、128 |
| `SFTC.EMBED_DIM`                | 768       |
| `SFTC.POS_SCALE`                | 0.1       |
| `SFTC.SOURCE_BIAS_INIT`         | −2.0      |
| `SFTC.SOURCE_BIAS_FIXED_EPOCHS` | 5         |
| `SFTC.NEW_LR_MULT`              | 3.0       |
| `SOLVER.MAX_EPOCHS`             | 120       |
| `SOLVER.GRAD_CLIP_NORM`         | 1.0       |

这些是建议的配置项名称，新增后应保证代码和配置字段完全对应。

原 `UAD.yml` 不修改，以便保留独立基线的准确复现条件。

# 九、DDP：保留原 UAD，只额外注意三个地方

这次不需要重新设计分布式训练。

### 9.1 保留原 PKM 采样和特征排列

原 UAD 以三模态样本组织训练，并根据特征排列计算 ProCA/GPD。

新增模态 ID 必须与每个 Rank 实际输入顺序一致。

如果配置为 RGB、IR、Thermal，模型内部可以统一映射为0、1、2；但映射关系应由模态名称生成，而不是在模型里永久写死。

### 9.2 沿用现有 DDP 和 AMP

官方代码已经包含 DDP 封装及 GPD 原型同步。

本次只需要确保：

- CNN 和局部 K/V 被正确注册为模型参数；
- 来源偏置五 Epoch 固定规则在所有 Rank 一致；
- 不在 DDP 包装之后动态替换来源偏置参数对象；
- 不增加额外手动梯度同步。

### 9.3 保持与基线相同的实际训练规模

原 UAD 的 `IMS_PER_BATCH` 是采样器的基础 Batch 配置，实际图像数还涉及三模态展开与 DDP 分配。

因此，比较 SFTC 与 UAD 时，应核对：

- GPU 数量；
- 每 Rank 的实际图像数量；
- 每模态样本数量；
- PKM身份分组；
- 实际 Optimizer Step 数量；
- 随机种子及相应训练配置。

不要为了节省新增 CNN 的显存，悄悄减少样本数或更换采样器。

# 十、训练与正式推理的一致性

这一点必须保留为硬性要求。

## 10.1 训练阶段

输入：

\\[ (x,m,y) \\]

执行：

\\[ (x,m) \rightarrow \text{ViT+SFTC} \rightarrow f\_{\mathrm{preBN}} \\]

然后：

\\[ f\_{\mathrm{preBN}} \rightarrow \text{原 UAD 损失} \\]

## 10.2 正式推理阶段

输入：

\\[ (x,m) \\]

执行完全相同的视觉编码结构：

\\[ (x,m) \rightarrow \text{ViT+SFTC} \rightarrow f\_{\mathrm{preBN}} \\]

再做原配置中的特征归一化和距离计算。

不能训练时使用 CNN 增强特征，推理时又只调用原 ViT 的未经增强 CLS。

需要区分诊断关闭模式和正式推理模式。

来源关闭模式只是用于同权重模型的计算正确性检查及训练后机制分析。

它不能恢复独立 UAD Checkpoint 的参数，也不能代替原 UAD 的正式结果。

# 十一、正式训练前必须通过的检查

只保留必要的工程检查，不增加新的研究实验。

| 检查项目         | 必须满足的条件                        |
| ------------ | ------------------------------ |
| CNN 张量尺寸     | F4、F8、多尺度融合和128个局部 Token 尺寸正确  |
| ViT Token 数  | Block 7 前129个，Block 7 后仍129个   |
| Attention 形状 | `[B,12,129,257]`               |
| 预训练加载        | 原 ViT QKV、MLP、位置编码等关键参数成功匹配    |
| 局部 K/V 初始化   | 首次复制正确；Checkpoint 恢复不重复覆盖      |
| 来源偏置         | Epoch 1–5 保持−2.0，Epoch 6后正常学习  |
| 梯度路径         | CNN、KV、ViT 均有有效且有限的梯度          |
| 全局裁剪         | 位于 AMP 反缩放之后，阈值仍为1.0           |
| 模态标签         | 训练、验证、测试映射正确                   |
| 损失接口         | CE、WRT、ProCA、GPD 的原特征输入位置不变    |
| 关闭等价性        | 同权重、`eval()` 下恢复原 Attention 计算 |
| 训练/推理输出      | 正式模式均使用768维 SFTC CLS           |

其中梯度检查的作用是排除断图、参数未注册等实现错误。

即使这些检查全部通过，也不意味着新增 CNN 已经学到有用身份特征。

# 十二、正式实验需要保留的最少量学习证据

不建议新增十余项消融，但必须避免再次出现“模型训练结束才发现新模块几乎没发挥作用”的情况。

建议记录以下关键指标。

| 指标                           | 用途                 |
| ---------------------------- | ------------------ |
| 来源偏置 \\(b\_{m,h}\\)          | 判断模态/Head 来源先验如何变化 |
| CLS Query 的 \\(\rho_L\\)     | 判断 CLS 是否直接读取局部来源  |
| Patch Query 的平均 \\(\rho_L\\) | 判断局部信息是否主要进入 Patch |
| CNN/KV 裁剪前梯度与实际参数更新          | 判断新分支是否获得有效学习机会    |
| 全局裁剪系数及频率                    | 判断梯度是否长期受到大幅缩放     |
| 最终 CLS 开关差异                  | 判断训练好的局部来源是否影响检索特征 |

来源概率质量必须在 Attention Dropout 前统计。

不应只报告全 Token 平均来源权重，因为 Patch Query 的读取可能掩盖 CLS 直接读取不足的问题。

### 推荐检查节点

在同一次正式训练中记录 Epoch 1、5、6–10和120附近的必要状态即可。

不需要在这些节点分别重新进行大规模实验。

### 重点避免三种错误判断

错误一： \\(\rho_L\\) 较高，因此 CNN 学到了身份信息。

不成立。它可能学习背景或重复主干信息。

错误二： CNN 参数变化明显，因此新模块有效。

不成立。参数更新量不代表优化方向对身份检索有利。

错误三： 同一 SFTC Checkpoint 开启局部来源优于关闭，因此 SFTC 必然优于 UAD。

不成立。SFTC 训练已经改变共享 ViT 权重，必须与独立训练的 UAD 进行正式比较。

# 十三、正式实验与结果判定

首轮只开展一项核心结构实验：

UAD Baseline vs UAD + SFTC-V2.1

其余训练协议保持相同。

重点报告：

| 评测内容                                 | 需要判断的核心问题      |
| ------------------------------------ | -------------- |
| 整体 AS-ReID mAP、Rank-1、Rank-5、Rank-10 | 完整方法是否改善统一检索   |
| TIR→TIR、NIR→NIR                      | 弱模态内部身份特征是否改善  |
| RGB/NIR↔TIR                          | 是否有利于跨模态对应     |
| RGB→RGB、NIR→NIR                      | 是否损害已有较强模态     |
| Ground→Aerial、Aerial→Ground          | 是否改善跨平台泛化      |
| 完整 Gallery 逐 Query AP / Rank-1 迁移    | 是否只修复近邻却损害整体排序 |

沿用原 UAD 的 Query/Gallery、相机过滤、`NECK_FEAT=before`、L2归一化设置，不启用 Re-ranking 或 TTA。

历史 UAD-DDP 指标可作为参考，但不能将不同训练配置和不同 Checkpoint 的结果混合比较。

### 如何解释可能的失败结果？

| 实验现象                     | 优先考虑的原因                       |
| ------------------------ | ----------------------------- |
| CNN 更新弱、来源权重低、CLS 开关差异极小 | 新分支可能被来源竞争和优化过程忽略             |
| 来源权重正常，但最终 CLS 差异极小      | 局部信息可能没有有效传递至最终身份表示           |
| CLS 变化明显，但整体 mAP 下降      | CNN 可能提取冗余或非身份信息，或替换了有益的主干注意力 |
| 整体提升主要来自 RGB             | 通用局部编码有益，但不能声称已解决 TIR 瓶颈      |
| TIR→TIR 改善，但跨 TIR 组合下降   | 模态内部细节可能有益，但统一跨模态对应受到影响       |

后续是否需要辅助监督、独立预训练或者更有针对性的弱模态提取机制，应根据这些证据再决定。

# 十四、最终交付与不得擅自修改的内容

开发完成后，建议要求实现者交付：

- 最终修改文件清单和配置文件；
- 实际模型结构、参数量、FLOPs/MACs、显存和推理耗时；
- 预训练权重加载及局部 K/V 初始化检查；
- 来源偏置5 Epoch固定与释放日志；
- 模态 ID 映射与训练/推理一致性检查；
- 梯度裁剪及新分支有效更新记录；
- 完整120 Epoch训练日志和 Checkpoint；
- 与独立 UAD 同协议的完整检索结果。

按当前结构估算，新增参数约144万，其中局部 K/V 约118万。这个数字属于设计估计，实际成本应以完成后的模型统计为准。

未经重新审查，不应在首轮实现中自行增加额外 Local CE、修改采样器、引入多模态专属 CNN、额外来源门控、第二个 CLS，或改变 UAD 的损失权重。

## 最终结论

SFTC-V2.1 可以正式实施。

其方法主线是：

> 针对 WHU-MARS 中共享 ViT 对不同光谱和平台下细粒度身份信息利用可能不足的问题，通过轻量 CNN 提取局部空间特征，并在共享 ViT 的第7个 Transformer Block 中引入来源校准注意力，使不同来源的信息参与统一身份特征形成。

本次设计已经从最初的“跨层残差补充”转向了“互补特征来源与统一编码”，同时解决了持续追加 Token、重复注入 CLS、预训练参数兼容、来源竞争定义、训练推理不一致以及新分支过早退出学习等主要设计风险。

但仍要保留一个清醒的判断：

固定偏置、提高新参数学习率、保留梯度裁剪以及使用 UAD 四项损失，只能提供合理的优化条件，不能保证 CNN 最终提取到不可替代的身份信息。

首轮最重要的目标不是证明 Attention 使用了 CNN，而是检验：

CNN 是否真正形成了有价值的身份特征增量，并最终改善弱模态与统一混合 Gallery 的检索性能。

