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
