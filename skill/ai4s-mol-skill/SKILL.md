---
name: ai4s-mol-skill
description: 在本实验仓复现真实分子清洗、三维构象、Uni-Mol 微调与 CUDA 空间组学基准，并输出可追溯指标和中文图表。用于运行或核对这条 AI for Science 流水线。
---

在仓库根目录操作。先读 `EXPERIMENT_PLAN.md` 与 `REPORT.md`，确认本机 `.venv/bin/python`、CUDA 和缓存权重可用。`REPORT.md` 是最近一次实测，不能当作未来运行结果。

执行入口：

```bash
.venv/bin/python run_pipeline.py
```

只运行指定模块可用 `--modules rdkit`、`--modules unimol` 或 `--modules cuspa`。Uni-Mol 依赖 RDKit 生成的数据和构象，先运行 rdkit。仅验证已有结果可显式使用 `--reuse-results`；此模式不产生新的训练/测速结果，manifest 会注明复用。

- RDKit：从真实 ESOL/BBBP 原始表开始，保留来源和失败计数。去盐并不等于完成质子化、互变异构与立体化学标准化。生成 ETKDGv3 + MMFF94 构象，记录未收敛与失败；不能把单个低能构象当成实验晶体结构。
- Uni-Mol：加载官方预训练权重，打印真实架构与参数。固定相同 train/validation/test 分组给所有模型；标准化仅拟合 train；按 validation 选 checkpoint，test 仅最终评估。保留逐 epoch loss 与时间，不能预设三维模型一定胜出。
- cuSpa：从本地官方 CUDA 源编译，运行明确标注的合成空间负载。CPU 与 GPU 使用同组多边形/点/基因，检查 assignment 和聚合结果。区分 GPU 核函数与端到端耗时、冷启动与稳态；不把合成细胞图称为患者数据。
- 内存：入口用 `guard_process.py` 每 250ms 检查统一内存，MemAvailable <20GiB 终止子进程组。不要同时运行两组 GPU 计时。
- 图表：只使用本次 results 指标；SVG 白底 `#ffffff`、WenQuanYi Micro Hei、单栏 560–680px。转 PNG 检查中文、裁切与重叠。药物结构性质只解释分子机制，不作临床用药结论。

输出保存在 `results/`、`logs/`、`visuals/`。将新命令、版本、计时口径、失败与修复按时间追加到 `DevHistory.md`，更新 `REPORT.md` 时保留实测与推断的区别。

## 第二阶段：血脑屏障、非键力场、蛋白质

读取 `PHASE2_FULL_AI4S_PLAN.md`。入口 `.venv/bin/python run_pipeline.py --phase 2` 按 `bbbp → forcefield → protein` 顺序执行；`--reuse-results`只验证既有产物，不制造新的训练与计时。`--phase all`串联两阶段。依赖现有AMD SSH别名`amd9070`及远端只读借用的HIP工具链，输出写入`C:\AI4S-HIP-Lab`，不修改Strata工程。

- BBBP：按canonical母核去重并明确冲突标签处理；stratified固定train/val/test。用validation ROC-AUC选轮次，test一次；PR-AUC必须说明使用average precision还是梯形积分。遮挡归因定义为原logit减扰动logit，按实际正负染色，不能先规定亲脂为正、氧原子为负。它是模型敏感性，不是临床因果证明。
- Pair-bias：使用真实checkpoint参数，包括128→128/GELU→64投影和真实首层QKV。明确输入是否已执行layer norm、padding/BOS/EOS掩码，以及核输出是bias、概率还是attention输出。跨GPU结果还要对独立PyTorch参考，不能只证明同一份错误在两块GPU一致。
- 非键力场：说明PBC最小镜像、单位、截断、反应场介电参数、LJ是否平移。27邻格要求每轴至少3格，避免周期重复。每原子遍历邻居时总能量除2、受力不除2；先做小体系bruteforce、解析力有限差分和总力检查，再跑32k/128k。报告是force evaluation/s，不把未跑的完整MD轨迹换算成ns/day。
- 蛋白质：真实实验CSV保留来源、哈希、去重和抽样协议；Biopython校验20种标准氨基酸。ESM-2按真实config打印词表与RoPE，训练和标签标准化均只用train。单点突变热图是模型预测变化，不假称已做湿实验。
- GPU计时依次运行，CPU准备可并行。本地重进程仍用统一内存看护。SVG禁止Unicode上标ᵀ，使用`^T`。

## 第三阶段：跨领域原型与完整Uni-Mol

`PHASE3_GRAND_SLAM_AMD_PLAN.md`对应`.venv/bin/python run_pipeline.py --phase 3`，GPU顺序为PME/MD→Davidson→FX→Profile-HMM→完整Uni-Mol。`verify_phase3.py`从原始数值复核；镜像仓仅保留轻量证据，不把摘要检查说成重新跑GPU。

- PME：Ewald实空间、倒易与自能口径一致；四阶B样条的能量和力用同一离散模型，另对直接Ewald、网格细化、有限差分及NVE步长减半。网格误差与跨GPU浮点误差分别报。NVT热浴不要求总能量守恒，无SETTLE就不宣称刚性约束。
- Davidson：保留完整H与独立精确解，只比较最后收敛迭代输出；同查本征值、H残差与正交性。CPU子空间/正交化和传输必须分账，不预设22倍，不把模型周期势称为硅自洽DFT。
- FX：电压协方差必须来自同一天空傅里叶可见度；恢复只读真实FX输出。标明合成天空、uv格权重、缺失DC、CLEAN迭代上限与固定恢复beam。FFT/X与CPU成像、I/O耗时分开。
- HMM：真实PSSM与合成植入/诱饵分开，gapless数据的I/D转移先验明说。用独立二维DP核对就地更新，全100k score/hash验证。glocal M/I/D原型不是官方HMMER校准分数或E-value。
- 全Uni-Mol：真实47,331,139参数、15层pair与node路径全导出，禁止用GB10预测作为AMD输入。保留旧BF16结果；同精度FP64逐层对齐，并额外验证官方FP32 Gaussian兼容路径对官方完整FP32的最大差<1e-5。不同精度指标不硬贴。

## 第四阶段：简化构象松弛、驻留求解与只读终审

`run_pipeline.py --phase 4`运行新构象优化和全GPU驻留求解，再核对已有381终审记录；最后一步不等于自动重新联网审稿。381正文/五SVG/封面只读，正式意见仅写`wechat/assets/381/审稿意见.md`；文件变化不能复用旧终审。

- 构象：必须fresh ETKDG起点，不能拿已优化构象冒充GPU松弛。明确简化bond/cos-angle/torsion/LJ+Coulomb参数及重原子近似；旧66.608秒是ETKDG+实际MMFF94，不与不同任务直接相除。Shared驻留仅指迭代状态，参数读取/诊断仍有global访存。
- 正确性不能只看能量下降与跨卡一致：还要看固定定义的碰撞和键偏差。本次过弱排斥失败原始数据要保留；改参数后不改起点/阈值，再做FD与CPU/CUDA/HIP全量复测。
- 驻留Davidson：共享Jacobi、两遍MGS和收敛控制均设备内；首次上传/最终下载另列，每轮不得偷偷回CPU解子空间。新seq/batch都驻留，倍数主要比较单k点block串行与多block占用率，不能冒称官方22倍复现。精确eigh和完整H残差要同时验收。
- 审稿结论和运行验收分开：审稿完成但有P0/P1，并不表示正文可发布；HTTP200验证码不等于读到微信正文。给出原句、数据/一级来源、局部建议，保留作者口气，不改正文替用户决定。
