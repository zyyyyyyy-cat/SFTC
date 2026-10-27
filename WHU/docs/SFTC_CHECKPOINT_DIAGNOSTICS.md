# SFTC Epoch 120：四项无需重训的验证

本实现遵循用户提供的针对性诊断指导。独立入口为 `WHU/diagnose_sftc.py`，无需运行训练，也不改变任何已训练权重值。

## 文件与原逻辑的关系

| 新增文件 | 功能 |
|---|---|
| `diagnose_sftc.py` | 加载一次完整 SFTC 权重；调度四项实验、固定样本、结果与缓存导出；检查参数/缓冲区前后哈希 |
| `utils/sftc_inference_probe.py` | 仅在诊断上下文中临时绑定 Block 7 Attention 前向；五种读取模式、FP32 来源统计及中间层 hooks；退出或异常时恢复 |
| `utils/sftc_probe_metrics.py` | 完整 Gallery 分块距离与排序、原相机过滤、逐 Query AP/CMC/INP、相似度直方图及 CSV |
| `tests/check_sftc_probe.py` | 无数据集的干预公式、原路径等价性、原评测对齐、导出和状态恢复自检 |
| `tests/check_sftc_probe_cli.py` | 构造临时合成 JPG 数据和完整权重，检查真实命令入口的四项输出、Query 索引及缓存复用 |
| `docs/sftc_probe_validation.json` | 本地合成检查结果；不是 WHU-MARS 的真实验证结果 |

`train.py`、`test.py`、`processor.py`、模型主文件、原评测器、采样器、损失、优化器和配置均不修改。原单卡及 DDP 训练入口保持原样。旧 `test.py` 的 `SFTC.TEST_LOCAL_ENABLED` 仍可单独验证 On/Off；独立诊断入口统一一次加载权重并导出四项实验所需文件。

诊断程序使用一个进程、一张可选 GPU。不用 `torchrun`，不创建 DDP，不创建训练 DataLoader/采样器，不使用 DataParallel。CPU 也可运行，但完整数据集建议使用 GPU。

## 运行命令

以下为 Linux 训练服务器示例，在 `WHU` 目录执行。把 Checkpoint 路径改成实际的第 120 轮 SFTC 权重文件；配置也可换成训练服务器上的 `configs/SFTC.yml`。

先执行实验 1、2：

```bash
CUDA_VISIBLE_DEVICES=0 python diagnose_sftc.py \
  --config_file configs/UAD_SFTC_V2_1.yml \
  --experiments 1 2 \
  --checkpoint /project/code/WHU/logs/SFTC_1000_DDP/transformer_120.pth \
  --output ./logs/SFTC_120_diagnostics \
  --device cuda:0 --batch-size 64 --query-chunk 32 \
  -- DATASETS.ROOT_DIR /project/code/data DATASETS.DATASET_DIR WHU-MARS
```

再执行实验 3、4，复用同一输出目录中已验证身份的特征缓存：

```bash
CUDA_VISIBLE_DEVICES=0 python diagnose_sftc.py \
  --config_file configs/UAD_SFTC_V2_1.yml \
  --experiments 3 4 \
  --checkpoint /project/code/WHU/logs/SFTC_1000_DDP/transformer_120.pth \
  --output ./logs/SFTC_120_diagnostics \
  --device cuda:0 --batch-size 64 --query-chunk 32 --samples-per-domain 50 \
  -- DATASETS.ROOT_DIR /project/code/data DATASETS.DATASET_DIR WHU-MARS
```

也可以把 `--experiments` 改为 `1 2 3 4` 一次完成。数据相关配置覆盖必须放在最后，推荐使用命令中的 `--` 分隔。

Windows PowerShell 可使用同样参数的一行命令，提前设 `$env:CUDA_VISIBLE_DEVICES='0'`；替换本机的权重和数据路径即可。

必须使用完整 SFTC 权重，不能用 UAD-only 权重。本程序复用现有严格 SFTC 加载器，加载完整推理参数；分类器沿原测试规则跳过，其不参与 pre-BN 检索。构造时不依赖额外 ImageNet 文件，完整已训练 Checkpoint 会覆盖全部实际使用参数，加载后不再复制 K/V 或初始化模型。

所有实验显式运行 `eval()`、冻结梯度、`inference_mode()`，不开 AMP、TTA 或 Re-ranking。要求 `SFTC.ENABLED=True`、`NECK_FEAT=before`、`FEAT_NORM=yes`、`TOP_K_EVAL=0`。`SFTC.TEST_LOCAL_ENABLED` 不选择实验模式；独立入口按实验名明确控制，原 `test.py` 的开关语义不变。

## 干预模式与特征位置

| 模式 | Block 7 的干预 |
|---|---|
| `on` | 与原正式测试相同的联合 Softmax、正常局部 K/V |
| `off` | 不执行局部编码器，不加入局部 K/V，与原关闭路径一致 |
| `zero_local_value` | 保留局部 Key、来源偏置及联合 Softmax；仅把局部 Value 聚合贡献设为零 |
| `cls_only` | 只有 CLS Query 开放局部候选；Patch Query 的局部 Logit 屏蔽为 -Inf |
| `patch_only` | CLS Query 的局部 Logit 屏蔽为 -Inf；Patch Query 开放局部候选 |

Query 屏蔽不改变 K/V 权重或来源偏置。屏蔽行仍保留原主干候选，归一化后与仅原主干候选计算一致；允许行保持联合来源计算。所有后续 Blocks 完全相同。

实验 4 的 `cnn_fused` 是 fusion Conv/GN/GELU 后 `[B,256,16,8]` 的空间平均，在 256→768 投影与二维位置编码之前提取；`local_value` 是训练完成的独立 V 投影后 `[B,128,768]` 的 Token 平均，与合并 12 Head 后再平均等价。两类特征均 L2 归一化，不增加或拟合分类头。

实验 3 每个模态×平台最多 50 张，先按身份轮转，再按路径 SHA256 确定图像顺序，避免仅选文件列表开头。固定列表保存为 `03_fixed_samples.csv`，涵盖原 Query/Gallery；缺失组合记录实际数量。四种模式始终使用相同图像与批次。

传播位置包括 Block 7 输入 CLS/Patch、Block 7 Attention 的输出 Projection/Dropout 后且残差相加前的 CLS/Patch、Block 7 输出 CLS/Patch、Block 8/10/12 的 CLS、最终 Norm 后的 CLS。Patch 差异使用每图展平的 `128×768` 向量。相对 L2 分母是 Off 范数加 `1e-12`，并逐批严格检查 Block 7 全部输入 Token 相同。

## 结果文件

| 实验 | 文件 |
|---|---|
| 1 | `01_retrieval_comparison.csv`、`01_query_comparison.csv`、`01_rank1_transitions.csv`、`01_manifest.json` |
| 2 | `02_source_decomposition.csv`、`02_query_comparison.csv` |
| 3 | `03_path_retrieval.csv`、`03_layer_propagation.csv`、`03_fixed_samples.csv` |
| 4 | `04_local_feature_retrieval.csv`、`04_identity_similarity.csv`、`04_feature_manifest.json` |
| 共同 | `attention_source_statistics.csv`、`evaluation_samples.csv`、`diagnostic_manifest.json`、`diagnostic_log.txt`、`features/` |

检索 CSV 含整体 AS-ReID、九种有向模态组合、A→G/G→A，每种模式 12 行，含 mAP、R1/R5/R10、mINP、Query/Gallery/有效 Query 数量及状态。

**检索汇总 CSV 指标为百分数 0–100；逐 Query AP/INP 为 0–1，R1 为 0/1，first_hit 是相机过滤后的一基排名。**没有合法同身份 Gallery 的 Query 保留稳定行与索引，但指标留空，不纳入均值。空协议也保留状态行；若 Gallery 总数少于某 CMC Rank，该 Rank 留空。

评测排除与 Query 同相机的全部 Gallery 样本，包括异身份，不只排除同身份同相机。平台沿原 WHU：零基 camid<5 为 G，>=5 为 A。各模式始终使用同模式的 Query 和 Gallery，查询顺序与原评测器按模态分离后合并的顺序一致。CSV 同时保留完整图像路径。

相似度统计使用完整 Query×Gallery 配对并排除同相机，区分同身份和异身份，按整体、九种模态组合、四种平台组合，以及 36 种模态×平台组合分别统计 count、mean、population std、min/max 和 40 档余弦直方图。组合统计不是相互独立的样本。后加的统计组合只累积距离分布，不额外排序，不计入 `04_local_feature_retrieval.csv` 的 12 个正式协议。

注意力统计按 mode、模态、平台、CLS/Patch、Head 保存图像均值。统计采用显式关闭 autocast 的 FP32，均为 detached、Dropout 前；两来源聚合响应范数与方向相似度位于输出 Projection 前。

`logZ_local_minus_backbone` 不含偏置，`effective_logZ_difference` 含实际已训练偏置。ZeroV 的实际 A_L 范数/余弦为零，同时保留 `A_L_original_*` 描述未清零候选内容；零向量余弦为程序的数值约定，不能解释为与 A_D 正交。Off 不执行局部来源，不伪造局部统计行。所有来源信息存聚合统计，不保存完整注意力矩阵。

## 缓存、资源与重跑

为了分两批运行时不重复提取 On/Off，完整特征保存在 `features/*.npy`，用 CPU memmap；每批都验证缓存中的 Checkpoint SHA256、配置、代码 SHA256 和包含全部路径/标签的元数据 SHA256。不同身份的缓存会拒绝复用，需选择新输出目录。

缓存不哈希每张图像内容，复用期间不要替换数据文件。未完成提取只留下 `.partial.npy`，完成后才改名；已有缓存还检查维度。重新计算可手动移走对应完整特征和其 attention JSON，或用新输出目录。

完整运行的特征磁盘容量约为 `(5×768+256+768)×图像总数×4` 字节。例如 185,922 张评测图像约 3.62 GB（十进制）；这是容量示例，不是当前数据规模断言。实验 1、2 仅三个最终 CLS 缓存，约 `3×768×N×4` 字节。距离和排序每次只处理 `--query-chunk` 张 Query，不保存完整距离矩阵；Gallery 特征仍须容纳于 CPU 内存。显存不足先减少 `--batch-size`，距离排序内存不足减少 `--query-chunk`，这些只改变计算分块，不改协议。

`diagnostic_manifest.json` 记录最近一次执行选项、检查和文件列表；第一批已有 CSV 保留。第二批输出与第一批共享严格校验的特征身份。全部结束时再次核验模型参数及缓冲区哈希。

## 本地自检与边界

```bash
python tests/check_sftc_probe.py
python tests/check_sftc_probe_cli.py
```

已检查：On/Off 与原路径精确相等；ZeroV 独立公式正确且与 Off 有区别；CLS/Patch Query 屏蔽正确；异常时恢复前向；训练模式拒绝诊断；autocast 下统计仍为 FP32；AP/CMC/mINP 与原 `eval_func` 对齐；无合法匹配 Query 的顺序保留；相同相机的异身份也排除；分块大小不影响结果；直方图计数正确；真实 SFTC 结构的五种前向、两个局部池化位置、240 行传播导出及全状态不变。

另外已用临时合成的 12 Query、12 Gallery JPG 和完整 SFTC 结构权重执行真实入口，四项实验全部导出成功，得到 480 行层传播、200 行身份相似度统计，并验证第二次运行复用特征、不重复提取。这里的随机权重与合成图片仅用于检查程序，不分析性能。8 个原核心文件的 SHA256 与修改前一致。

本地没有真实 WHU-MARS 数据和已训练 Checkpoint，无法提前产生实验指标或验证服务器耗时。上述 JSON 只证明合成实现检查通过，不代表任何实验结果。
