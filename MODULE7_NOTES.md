# 模块七：原创开源非键力场 CPU / CUDA / HIP 实测

源码 `amd_hip_port/cupotential_open/open_nonbonded.cpp`，驱动 `exp5_open_cupotential_hip.py`，数据与逐轮计时 `results/module7_results.json`，图6 `visuals/fig6_open_cupotential_md_forcefield_amd_vs_nvidia.svg`。图已转PNG并目检，640宽白底、文泉驿中文、无重叠/缺字。

## 物理实现与边界

原创MIT同源C++，分别编译单线程CPU、sm_121 CUDA、gfx1201 HIP。**没有调用或复刻NVIDIA libcupo二进制，不宣称兼容其API，更不称完整替代cuPotential。** CUDA/HIP仅使用runtime+CUB/hipCUB排序扫描，额外依赖仍遵守各自许可。

- 合成中性总电荷体系，32,000/128,000原子，种子20261123。坐标为3.5Å格距±0.15Å扰动，最小间隔下界3.2Å；电荷均匀±0.15e后去平均，sigma2.8–3.2Å，epsilon0.12–0.28kJ/mol。不是实验轨迹、不是带真实化学拓扑的溶剂模型。
- 三维立方PBC、最小镜像 `d-=L*floor(d/L+0.5)`、rc10Å。每格宽≥rc，网格每轴≥3，27邻格无周期重复。GPU分桶、排序、扫描、LJ/RF融合受力全部真运行在GPU；没有CPU邻居fallback。
- LB混合 sigma平均、epsilon几何均值。LJ12-6加reaction-field Coulomb；k_e=1389.354576 kJ Å/(mol e²)，RF外介质相对介电常数78.3。
- 对 d=ri-rj，F系数为 `24eps(2sr12-sr6)/r² + keqiqj(1/r³-2krf)`。每个i扫所有j，能量除2，力不除2；输出每原子能量与三维受力，避免浮点atomic reduce。
- 单位坐标/sigma为Å，能量kJ/mol，受力kJ/(mol Å)。**LJ未平滑、未shift，硬截断存在势能/力不连续**；RF势能在rc归零，但有限介电RF的力在rc不严格归零。没有键、1–4缩放、PME、积分器、温控压控。因此只报告非键评估次数/秒，不报MD step/s或ns/day。

## 独立验证：不是“共同写错也对上”

216原子33Å盒、rc10Å、3³格，C++ cell list对**独立NumPy全对全O(N²)表达式**：

- 总势能 -22.43521473139709 kJ/mol，相对误差1.58354e-16；全部force相对L2 3.12674e-16。
- 六个不同原子/坐标分量做中心有限差分，势能由独立NumPy暴力实现计算（不是调用共享C++pair helper），步长1e-4Å，距离截断最小余量0.103Å，无cross-cutoff问题；最大绝对力误差1.05155e-10 kJ/(mol Å)。
- 净力L2=2.82016e-15；跨盒周期平移后全部force相对L2=1.23016e-15，势能绝对变化2.13e-14。
- 完整验证记录 `results/module7_cpu_validation.json`。

32k与128k三后端输出的**所有per-atom energies和全部3N force分量均逐位一致**，不是仅比较总和，目标relativeL2<1e-12通过，实测0。各后端timings JSON总势能的末几位有Linux80-bit long double和Windows64-bit long double串行累计差异，最高约1.1e-14相对误差；统一由NumPy对相同energy数组归约后总势能完全相同。结果JSON同时保留原始host总能量与canonical NumPy总能量，未隐藏归约口径差异。

## 真实计时（同步host wall中位数，毫秒）

| N | 后端 | 索引 | 力+能量 | 合体流程 |
|---:|---|---:|---:|---:|
| 32,000 | CPU单线程C++ | 0.081233 | 61.084684 | 61.168222 |
| 32,000 | NVIDIA GB10 | 0.073265 | 7.921072 | 7.992897 |
| 32,000 | AMD RX9070XT | 0.177300 | 7.901600 | 7.901000 |
| 128,000 | CPU单线程C++ | 0.503209 | 241.310300 | 242.566880 |
| 128,000 | NVIDIA GB10 | 0.087713 | 27.104086 | 27.187688 |
| 128,000 | AMD RX9070XT | 0.270500 | 23.112800 | 22.426400 |

CPU3次、GPU7次采样，另2次预热。每段同步计时，合体整体独立测量，因此阶段中位数之和不等于合体中位数（GPU热态、时钟也有波动）。GB10/AMD的float64低于其FP32/AI吞吐率，**这里没用Tensor Core，也不应把本结果外推为所有分子力场性能排名**。

设备驻留基准不含输入输出：32k CUDA H2D/D2H=0.059809/0.045872ms，HIP=7.9122/0.9681ms；128k CUDA=0.146930/0.096977ms，HIP=8.0193/1.3610ms。传输是单次测量，不是中位数。只评估固定坐标，**没有模拟动力学轨迹**。

## 顺序、工具与踩坑

1. 按模块六→七→八：模块六代理确认BBBP训练+pairbias两端均结束后，才启动此模块GPU；全部CUDA/HIP结束并通过校验后，已立即通知模块八蛋白质训练代理放行。
2. 所有本地CPU/GPU执行都由250ms统一内存guard包裹，20GiB阈值。CUDA两负载最低可用109.65/109.87GiB，全部退出0，未触发熔断。CPU准备阶段有其他内存工作，最低仍100GiB。
3. GPU预分配scratch与输出，CUB/hipCUB稳定排序保留atom原序；关闭FMA contraction且不启fastmath。单线程CPU使用同序稳定计数分桶；独立暴力与有限差分另外验算，避免同源验证闭环。
4. Windows在 `C:\AI4S-HIP-Lab\cupotential_open` 独立目录，仅读现有TheRock ROCm/VS环境，未改Strata。首次HIP编译直接成功，仅析构hipFree未检查返回值的nodiscard警告，完整日志随源码保留。
5. 复现：`.venv/bin/python exp5_open_cupotential_hip.py` 全流程；`--prepare-only`仅CPU物理验证与数据；`--gpu-only`用准备好的输入跑两GPU；`--summarize-only`只读已有真实结果重出JSON/图。脚本内部每个本机重进程都自动guard。
