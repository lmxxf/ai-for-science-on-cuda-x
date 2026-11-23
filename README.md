# AI for Science on CUDA-X：从科学库 (`RDKit`)、专有模型微调 (`Uni-Mol`) 到 `CUDA-X` (`cuSpa` / `cuPotential`) 与科研 Skill 串联实战

本项目在 **`NVIDIA DGX Spark`（GB10，128GB 统一内存，CUDA 13.0）** 上，完整跑通并解剖了现代 **AI for Science（科学智能体）** 的四层技术栈：
1. **科学计算库与实验数据处理（`RDKit`）**：真实药物数据集 `ESOL`（1,128 分子水溶性）与 `BBBP`（2,050 行拦截 11 条非法结构得 2,039 分子血脑屏障穿透率）的化学价校验、`.Cl` 去盐清洗、`MW / LogP / TPSA` 理化描述符计算，以及 `ETKDGv3 + MMFF94` 力场生成 3D 原子坐标与三维距离矩阵 `D[i, j]`（抓出降压药**硝苯地平**在 2D 图上相隔 7 根化学键的两个氧原子在 3D 空间折叠至 **`3.453 Å`** 的真实案例）；
2. **3D 分子专有大模型全参数微调（`Uni-Mol v1`）**：加载官方 **`4,733 万`参数（`47,331,139` 参数，15 层，64 注意力头，128 个高斯核 3D 距离偏置）** 预训练权重，在去重母核后的 `ESOL` 同考卷上跑完 20 轮全参数微调（GPU 仅耗时 **`37.18 秒`**），将测试集预测误差 `RMSE` 从 `RDKit` 2048 维 2D 指纹+随机森林的 **`1.144`（`R² = 0.723`）** 一刀砍半至 **`0.574`（`R² = 0.930`）**；
3. **NVIDIA SC26 `CUDA-X` 科学库源码审计、真机编译与 `AMD RX 9070 XT (HIP)` 跨架构移植对账（`cuSpa` & `cuPotential`）**：
   - 用 `nvcc 13.0` 在 `DGX Spark (NVIDIA GB10)` 上原生编译刚开源的 **`2,378` 行**纯 CUDA 空间单细胞库 `NVIDIA/cuspa`（二维网格包围盒过滤 + 128 行水平射线法 `is_point_in_polygon` + 64 位整数打包 `CUB` 基数排序转 CSR 稀疏矩阵）；
   - **真机移植至 `AMD Radeon RX 9070 XT`（`gfx1201` RDNA4，原生 Windows ROCm 10.2 `HIP + hipCUB`，代码位于 [`amd_hip_port/`](./amd_hip_port/)，独立镜像仓 [`lmxxf/ai-for-science-on-amd-9070xt`](https://github.com/lmxxf/ai-for-science-on-amd-9070xt)）**：在同一份 `50,000 细胞 × 2,000,000 转录点 × 500 基因`（`float64`，种子 `20261123`）二进制输入上：
     - **性能**：同源 C++ 入口下，**`AMD RX 9070 XT` 设备驻留全流程耗时仅 `6.050 ms`**（对比 Numba 编译后 CPU 的 `272.274 ms` 提速 **`45.0×`**，与 `NVIDIA GB10` 同源 C++ 入口的 `5.760 ms` 基本持平；其中 **200 万点水平射线几何查询在 `AMD RX 9070 XT` 上仅需 `0.921 ms`**，比 `GB10` 的 `1.520 ms` 更快，比编译后 CPU 的 `236.064 ms` 快 **`256.3×`**）；
     - **逐字节零误差（Bit-Exact）**：`AMD RX 9070 XT` 与 `NVIDIA GB10` 产出的 2,000,000 个点归属数组及 `50,000 × 500` CSR 稀疏矩阵（`indptr / indices / data` 共 780,563 个非零项）**四个数组的 SHA256 哈希值 100% 逐字节完全相同（`0 mismatch`）**！
   - 同时厘清 `NVIDIA/cuPotential` 开源 OpenMM 融合积分器插件与闭源 `libcupo.so` 核心运行时的边界；
4. **科研 Agent `Skill` 工具链串联**：提供标准 [`skill/ai4s-mol-skill/SKILL.md`](./skill/ai4s-mol-skill/SKILL.md) 与带 250ms 统一内存安全看护的一键流水线 [`run_pipeline.py`](./run_pipeline.py)。

- **完整实测报告与工程日志**：[`REPORT.md`](./REPORT.md) ｜ [`DevHistory.md`](./DevHistory.md) ｜ [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md)

---

## 🖼️ 四张核心实测可视化大图

| 模块一：`RDKit` 去盐清洗与一二代抗过敏药极性对比 | 模块一：硝苯地平从 1D 字符串到 3D 折叠距离矩阵 (`3.453 Å`) |
| :---: | :---: |
| ![fig1](./visuals/fig1_rdkit_cleaning_and_bbb_drugs.svg) | ![fig2](./visuals/fig2_1d_smiles_to_3d_conformer_distance.svg) |
| **模块二：`Uni-Mol` 3D 微调 (`RMSE 0.574`) vs 2D 指纹 (`1.144`)** | **模块三：`cuSpa` 细胞多边形射线法 (`156.1×`) 与 AMD 移植审计** |
| ![fig3](./visuals/fig3_unimol_3d_finetune_vs_baselines.svg) | ![fig4](./visuals/fig4_cuspa_cell_polygon_raycasting.svg) |

---

## 🚀 快速复现

```bash
# 1. 仅校验已落盘的全套实测 JSON、数据切分无重叠与 SVG 图表（秒级完成）：
.venv/bin/python run_pipeline.py --reuse-results

# 2. 从头完整重跑 RDKit 清洗 + Uni-Mol 20 轮 GPU 微调 + cuSpa 原核基准测试（带 250ms 统一内存看护）：
.venv/bin/python run_pipeline.py
```

## 第二阶段：BBBP、开放非键力场、蛋白质 ESM-2

第二阶段已完成真实实验，设计见[PHASE2_FULL_AI4S_PLAN.md](PHASE2_FULL_AI4S_PLAN.md)，完整数据口径见[REPORT.md](REPORT.md)。

- BBBP同测试集：Uni-Mol ROC-AUC0.93333，Morgan RF0.93261，基本打平；真实逐重原子遮挡归因与官方权重Gaussian/首层注意力CUDA-HIP核通过参考验证。
- 原创MIT非键力场：PBC最小镜像、27邻格、LB混合LJ12-6、反应场库仑与解析力。128k原子CPU242.57ms / GB1027.19ms / RX9070XT22.43ms；完整能量/受力数组跨后端一致，暴力参考与有限差分通过。非完整MD引擎，不兼容或复刻专有libcupo。
- 官方ESM-2 7.51M全参数微调真实GB1：固定3000条实验序列、2400/300/300；ESM测试RMSE0.538，425维生化特征RF0.870，Spearman0.898。突变热图是模型预测，不冒称新湿实验。

```bash
.venv/bin/python run_pipeline.py --phase 2
.venv/bin/python run_pipeline.py --phase 2 --reuse-results
.venv/bin/python verify_phase2.py
```

| BBBP / 原子归因 / AMD注意力核 | 开源非键势能与受力 | ESM-2与单点突变扫描 |
|---|---|---|
| ![fig5](visuals/fig5_bbbp_unimol_roc_and_atom_attribution.svg) | ![fig6](visuals/fig6_open_cupotential_md_forcefield_amd_vs_nvidia.svg) | ![fig7](visuals/fig7_protein_esm2_finetune_and_mutation_scan.svg) |
