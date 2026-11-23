# AI for Science 实战实验图纸：从 NVIDIA CUDA-X (`cuSpa` / `cuPotential`)、科学库 (`RDKit`) 到深势 3D 分子专有模型 (`Uni-Mol`) 微调与科研 Skill 串联

**实验目录**：`/home/lmxxf/work/ai-for-science-on-cuda-x`  
**创建时间**：2026-11-23 22:34 (+08:00)  
**主脑规划**：枢木朱雀（鳩 / `Gemini 4 Argon`）  
**工程执行**：闇（`GPT-6.1-Sol low`）  
**目标读者**：让完全没接触过计算化学/空间生物学的程序员，通过**直观的 SVG 可视化图 + 真机实测秒表与准确率表格**，10 分钟看懂：
1. **科学库（`RDKit`）到底在干啥**？（什么是 `SMILES` 字符串、怎么洗掉实验脏数据/盐酸盐、怎么算 2D 理化指标、怎么把 1D 字符串折叠成 3D 空间原子坐标 `(x, y, z)`，为什么 CPU 捏 3D 分子构象是喂给 GPU 模型前的最大耗时瓶颈）；
2. **分子专有模型（深势科技开源的 `Uni-Mol` 3D Transformer）微调到底在干啥**？（为什么它只有 `84M` 参数，却能靠 **3D 原子距离矩阵注意力** 在药物性质预测［如血脑屏障穿透率 `BBBP` / 水溶性 `ESOL`］上吊打传统 2D 指纹和通用大模型）；
3. **老黄刚在 SC26（11-16）发布的 `CUDA-X` 科学库（`cuSpa` / `cuPotential`）底下到底是什么**？（扒开 `cuPotential`“只开源 OpenMM 插件壳子、核心 `libcupo.so` 闭源”的套路；实测编译并跑通 `2,378` 行纯 CUDA 的 `cuSpa` 细胞多边形 `Point-in-Polygon` 核函数，测出 CPU vs GPU 真实加速比，并画出直观的“细胞多边形 + 基因点射线判定”可视化 SVG）；
4. **科研 `Skill` 是怎么把它们串起来的**？（写一个干净可执行的 `ai4s-molecule-pipeline/SKILL.md` 及 CLI 驱动脚本，把“数据清洗 $\rightarrow$ 3D 构象生成 $\rightarrow$ `Uni-Mol` 微调 $\rightarrow$ 候选分子高亮出图”串成一条龙）。

---

## ⚠️ 铁律与安全红线（闇执行必读）

1. **统一内存安全看护**：`DGX Spark` 为 128GB CPU+GPU 统一内存，跑 GPU 训练/测试时注意控制 `batch_size` 与 worker 数（`num_workers=0` 或 `2`），防止内存泄漏；
2. **独立干净虚拟环境**：在 `/home/lmxxf/work/ai-for-science-on-cuda-x/.venv` 下建虚拟环境（或复用 `/home/lmxxf/work/interp-curvature-exp/venv` 的基础 torch 并使用 `--system-site-packages`），安装 `rdkit`、`unimol_tools`（或直接用 PyTorch 实现/加载 `Uni-Mol` 权重微调）、`scikit-learn`、`pandas`、`matplotlib`、`cairosvg`；
3. **全程记录 `DevHistory.md`**：在 `/home/lmxxf/work/ai-for-science-on-cuda-x/DevHistory.md` 中按时间正序详实记录每一步环境配置、踩坑修复、实验命令、真实耗时与输出指标，严禁任何 `"simulated"` 假数据；
4. **可视化 SVG 遵循公众号纯白底铁律**：所有生成的展示用 SVG（存放在 `/home/lmxxf/work/ai-for-science-on-cuda-x/visuals/`）必须垫 `#ffffff` 纯白底、顶层写 `font-family="'WenQuanYi Micro Hei','Noto Sans CJK SC',sans-serif"`、宽度 `560~680px` 单栏大字排版，并用 `cairosvg` 转 PNG 校验无乱码、无文字重叠。

---

## 实验分步执行清单（四大模块）

### 模块一：科学库 `RDKit` 实验数据处理与 2D/3D 直观可视化（`exp1_rdkit_data_processing.py`）
1. **真实数据集准备**：
   - 下载 MoleculeNet 标准真实药物数据集 **`ESOL`（`delaney-processed.csv`，1,128 个分子的实测水溶性 `logS`）** 和/或 **`BBBP`（2,039 个真实药物分子的血脑屏障穿透率 0/1 标签）**（加上几个大家耳熟能详的常见药做具象展示锚点：**阿司匹林 Aspirin**、**咖啡因 Caffeine**、**布洛芬 Ibuprofen**、**氯雷他定 Loratadine［不困的第二代抗过敏药，难穿血脑屏障］**、**苯海拉明 Diphenhydramine［会犯困的第一代抗过敏药，易穿血脑屏障］**、以及一条带盐酸盐杂质的原始实验记录如 `CN(C)CCOC(c1ccccc1)c2ccccc2.Cl` 盐酸苯海拉明）。
2. **`RDKit` 四步科学数据处理实测与计时**：
   - **Step 1（合法性校验与去盐清洗 Sanitization & Salt Stripping）**：展示原始实验表里的非法化学价字符串如何被 `Chem.MolFromSmiles` 拦截，带 `.Cl` / `.Na` 盐杂质的药物分子如何用 `SaltRemover` 剥离成纯净母核；
   - **Step 2（计算 2D 理化描述符 Descriptors）**：用 `RDKit` 算出分子量 `MW`、脂水分配系数 `LogP`（越亲脂越容易钻过大脑血管壁）、拓扑极性表面积 `TPSA`（极性太强 `>90 Å²` 就钻不进大脑！对比“犯困过敏药苯海拉明 `TPSA=12.5 Å²` 秒进大脑” vs “不困过敏药非索非那定/氯雷他定”）；
   - **Step 3（1D 字符串吹气变成 3D 空间原子坐标：`ETKDGv3` + `MMFF94` 力场优化）**：
     - 对 1,000+ 个分子调用 `AllChem.EmbedMolecule`（ETKDG 距离几何算法生成初始 3D 坐标）+ `AllChem.MMFFOptimizeMolecule`（经典分子力场能量最小化）；
     - **精准掐表计时**：统计单核 CPU vs 多进程 CPU 处理 1,000 个分子（每个分子生成 1 个和 10 个 3D 构象）的真实耗时，以及算出的 **3D 原子间欧氏距离矩阵 `D_ij ∈ ℝ^{N×N}`**（找出一个“在 2D 拓扑图上隔了 5 根化学键、但在 3D 空间里折叠过来距离只有 `< 3.5 Å`”的真实分子案例！）。
3. **生成可视化大图 1 & 2（存入 `visuals/`）**：
   - **`visuals/fig1_rdkit_cleaning_and_bbb_drugs.svg`**：用 `RDKit` 的 `MolDraw2DSVG` 嵌入组合出一张纯白底中文解读图——上半部展示“原始带盐酸盐 `.Cl` 的实验数据 $\xrightarrow{\text{RDKit 去盐清洗}}$ 纯净分子结构”；下半部并排对比 **“为什么第一代抗过敏药（苯海拉明）吃完直犯困，而第二代抗过敏药（非索非那定/西替利嗪）不犯困？”**（直接画出两者的 2D 分子结构，高亮标出极性基团 `-COOH / -OH`，列出 `RDKit` 算出的 `TPSA` 极性表面积和 `LogP`，让外行一眼看懂化学结构怎么决定药能不能进大脑！）；
   - **`visuals/fig2_1d_smiles_to_3d_conformer_distance.svg`**：左边是 1D `SMILES` 字符串，中间是 2D 平面骨架，右边是 `RDKit ETKDG+MMFF94` 算出的 **3D 立体球棍投影 + 原子 `(x,y,z)` 坐标与 3D 距离矩阵热力格**（高亮标出那对“2D 看着远、3D 折过来挨着”的原子对！）。

---

### 模块二：深势科技 3D 分子专有模型 `Uni-Mol` 真机微调对决（`exp2_unimol_finetune.py`）
1. **拆开深势 `Uni-Mol` 的模型结构（写进报告，像拆 LLM `config.json` 一样清楚）**：
   - 用代码打印出 `Uni-Mol` 的真实层数、隐藏维度、参数量（`~47M / 84M`），以及它和普通文本 Transformer 唯一的核心区别：**输入 Token 是原子种类 `Atom Type`，而每一层 Attention 的 `Q·Kᵀ / √d` 后面直接加了一项经过高斯核（Gaussian Kernel）展开的 3D 原子对欧氏距离偏置 `Bias(d_ij)`**！
2. **同考卷三方真机 PK（在 `ESOL` 水溶性回归 & `BBBP` 血脑屏障穿透分类两个真实数据集上，固定同一个 Scaffold/Random 留出测试集）**：
   - **选手 1（不微调的文本/简单基线或零样本特征）**；
   - **选手 2（传统化学库天花板：`RDKit` 2048 维 Morgan 2D 分子指纹 + `RandomForest / Ridge / MLP`）**：只有 2D 拓扑连接，没有 3D 空间构象；
   - **选手 3（深势 3D 分子专有模型微调：`Uni-Mol` 预训练权重 + 在我们的 `DGX Spark` GPU 上真机 Fine-tune）**：
     - 记录每一个 Epoch 的训练 Loss、验证集 `RMSE / MAE / R²`（对 ESOL）和 `ROC-AUC / Accuracy`（对 BBBP），以及 **微调前的阶段一（CPU `RDKit` 生成 3D 构象耗时） vs 阶段二（GPU `Uni-Mol` Transformer 微调前向+反向传播耗时）** 的真实秒数对比！
3. **生成可视化大图 3（存入 `visuals/`）**：
   - **`visuals/fig3_unimol_3d_finetune_vs_baselines.svg`**：展示 ① `Uni-Mol` 3D 距离矩阵注入 Attention 的架构简图与真实测试集预测散点/AUC 对比；② **耗时分账条形图**（直观展示为什么 GPU 训 10 轮模型很快，反而是前面 CPU `RDKit` 捏 3D 构象占了大头，引出为什么需要 GPU 加速科学库！）。

---

### 模块三：扒开老黄 SC26 `CUDA-X` 科学库（`cuSpa` 真机编译实测 + `cuPotential` 源码解剖，`exp3_cudax_cuspa_benchmark.py`）
1. **`cuPotential` 解剖对账**：
   - 核实 `/home/lmxxf/work/ai-for-science-on-cuda-x/cuPotential` 的真实结构：统计开源部分（OpenMM 插件 C++/CUDA 胶水层）与闭源部分（`libcupo.so` 预编译二进制依赖），并分析 `plugins/openmm/cpp/platforms/cuda/src/` 下它实际做了什么算子融合（如 Fused Integrator / Nonbonded 力场）。
2. **`cuSpa`（`2,378` 行纯 CUDA 空间生物学多边形库）在 `DGX Spark` 上从源码编译与真机跑分**：
   - 在 `/home/lmxxf/work/ai-for-science-on-cuda-x/cuspa` 下用本地 `nvcc`（CUDA 13.0）编译安装 `cuspa`（或直接用 `nvcc` 编译其核心 `index.cu + query.cu + aggregate.cu`）；
   - 构造一组真实的单细胞空间转录组规模测试（例如：`50,000` 个细胞多边形 × `2,000,000` 个基因转录本坐标点 `(x, y)` × `500` 种基因）：
     - 对比 **① CPU Python/Shapely 或 C++ 射线法耗时** vs **② `DGX Spark` GPU 运行 `cuSpa`（2D 网格包围盒索引 + 射线法 `is_point_in_polygon` + `atomicAdd` 稀疏矩阵聚合）的毫秒级耗时**，验证老黄说的“几百倍加速”到底是怎么来的，并确认其 `.cu` 代码 100% 不依赖 Tensor Core 或 NVIDIA 私有指令（为下一步 `hipify` 到 `AMD RX 9070 XT` 做好准备！）。
3. **生成可视化大图 4（存入 `visuals/`）**：
   - **`visuals/fig4_cuspa_cell_polygon_raycasting.svg`**：画出一块显微镜下的“肿瘤组织细胞多边形网格 + 落入其中的红蓝基因转录点 + `cuSpa` 的两步法（先按浅灰网格外接框过滤，再从基因点发射一条水平射线数穿过几次细胞边界）+ CPU vs GPU 实测耗时对比”。让你一眼看懂什么叫“空间单细胞多边形计算”！

---

### 模块四：科研 `Skill` 串联与总结报告（`skill/ai4s-mol-skill/SKILL.md` & `REPORT.md`）
1. 编写 `/home/lmxxf/work/ai-for-science-on-cuda-x/skill/ai4s-mol-skill/SKILL.md` 及一键入口脚本 `run_pipeline.py`：演示 Agent 如何通过一条工具调用串联 **`RDKit 数据清洗去盐 → 3D 构象生成 → Uni-Mol 专有模型微调/推理 → 输出分子结构高亮 SVG 报告`**；
2. 将全部实验硬数据、代码路径与 4 张可视化图汇总写入 `/home/lmxxf/work/ai-for-science-on-cuda-x/REPORT.md` 与 `DevHistory.md`。
