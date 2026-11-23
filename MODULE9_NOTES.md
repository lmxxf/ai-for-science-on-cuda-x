# 模块九：PME + 真实 NVE/NVT 动力学闭环

文件入口：`exp7_pme_and_fused_md_hip.py`、`amd_hip_port/cupotential_pme_md/pme_md.cpp`，汇总`results/module9_results.json`，图8`visuals/fig8_pme_kspace_and_1000step_md_trajectory.svg`。原创MIT，不称复刻libcupo、SETTLE、硬约束或真实实验轨迹。

## 完整实现了什么

81个原子构成27个中性柔性三原子合成分子；24Å周期盒，电荷(+.08,-.16,+.08)e，质量(12,16,12)amu，键长2Å/k200，角1.9rad/k30；全原子对LJ sigma1.2Å epsilon.05kJ/mol，无隐含键排除。

64³四阶B样条铺电→完整复数3D FFT（cuFFT/hipFFT）→Ewald格林函数、每维sinc^8反卷积→逆FFT→B样条导数插值势能/力。不是拿网格演示图冒充计算。电荷mesh、坐标、速度、力、能量诊断全设备驻留；每步真实重算PME。为保持同一离散模型，能量是1/2 rho·phi，力是B样条铺展权重导数，**没有混搭ik电场力导致能量梯度不一致**。

另有实空间MIC erfc(alpha*r)/r、LJ、Ewald自能、谐振键长及键角力。alpha=.4Å^-1、rc10Å；小体系实空间用全对全，未声称已有大型cell-list MD优化。LJ硬截断未shift；erfc在rc仍有极小尾。周期中性体系k0置零为tin-foil边界，无表面项。温度/积分单位：K=.005 sum m v²，a=100F/m，时间ps、长度Å、质量amu、能量kJ/mol。

## 独立Ewald与能量梯度

Python直接Ewald不走网格：实空间显式相邻周期像盒求和、倒易整数向量k∈[-16,16]³（再与18对照）、高斯自能，另加相同LJ/键/角模型。倒易截断收敛force相对L2=2.25e-15，真实输出在`module9_validation.json`。

| 网格 | 对直接Ewald全受力相对L2 |
|---:|---:|
| 32³ | 约4.64e-4 |
| 64³ | 约4.94e-5 |
| 128³ | 约5.84e-6 |

**64³不是对连续Ewald的机器精度**。128³改善约8.45倍。势能有限差分与同一mesh解析force最大绝对误差4.81e-9 kJ/(mol Å)。64³初始净力残差约3.82e-4 kJ/(mol Å)，来自有限mesh离散破坏连续平移不变性（32³约7.60e-3，128³约1.09e-4）；未把它说成严格零净力。

## 真实动力学、误差与计时

每后端NVE1000步dt0.5fs至0.5ps，半步长2000步dt0.25fs至同一0.5ps；NVT BAOAB1000步dt1fs至1ps，gamma10ps^-1，300K。NVT哈希/Box-Muller随机数随step/坐标确定，三平台使用同一随机驱动。GPU循环每步无CPU数据回传；融合kick/drift/thermostat核+PME核/FFT组合，**不是单个持久核，也不是SETTLE或生产级积分器**。

| 后端 | NVE1000步(s) | 半步长2000步(s) | NVT1000步(s) |
|---|---:|---:|---:|
| CPU NumPy/SciPy单FFT worker | 2.872697 | 5.749312 | 2.872045 |
| NVIDIA GB10 CUDA | 1.062893 | 2.124769 | 1.066722 |
| AMD RX9070XT HIP | 0.796764 | 1.347705 | 0.657632 |

单次完整轨迹wall，非重复benchmark中位数。CPU是Python/NumPy/SciPy基线，与GPU实现语言/调用不同，不能把比值解释为厂商产品加速倍数。设备初始化、初始输入传入、FFT plan创建、最终回传不计入轨迹循环；GPU每步诊断kernel开销计入。

NVE最大|ΔE|/|E0|=1.09019e-4；半步长2.72519e-5，改善4.0004倍，符合Velocity-Verlet二阶收敛。初始动能按300K设置、键角从平衡构型起步，NVE温度在动力学中向252K附近再分配，不能拿NVE温度不等300K误判温控。

NVT后半段平均297.3492K、标准差27.5717K；这是短1ps轨迹与自相关样本统计，不是证明生产级热力学采样收敛。NVT总能量不守恒是热浴交换，结果保留其实际变化而不当能量漂移失败。

跨CUDA/HIP初始force L2=4.66e-16，二者对CPU约4–5e-16；完整1000/2000步最终坐标对CPU最大MIC误差约2.5e-14Å，结果JSON逐轨迹给出。三个后端保存每步的势能/动能/总能量/T/首个键长/atom0-x历史，以及全原子最终XYZ和速度；**未声称存了全原子每步XYZ文件**。

## 工程、图与复现

源码完整说明、单位与编译方法见`amd_hip_port/cupotential_pme_md/README.md`。CUDA链接cuFFT、HIP链接hipFFT，原有ROCm/VS只读，独立远端子目录。double、禁fastmath、禁FMA contraction。实际C++模板源码在两个GPU相同；CPU独立实现且另有directEwald，避免共同错误闭环。

两GPU全轨迹完成/验证后已放行模块10，不再占GPU。CPU与CUDA均用250ms/20GiB统一内存guard。图8由真实铺电FFT切片与实测轨迹/计时生成，640白底文泉驿，已PNG目检无缺字重叠。
