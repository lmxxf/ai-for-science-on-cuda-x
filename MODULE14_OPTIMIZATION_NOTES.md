# 构象松弛实现优化：同一力场、相同轨迹，热态真机对照

日期：2026-11-25。只优化执行方式，没有换输入、力场、epsilon、精度、步数或回溯策略。主module14_results.json与DevHistory均未改，主合并由父代理处理。

## 先保全原证据

原source、exp12驱动、Windows构建/运行脚本、旧结果JSON及CPU/CUDA/HIP完整坐标/201步历史均存于`history/conformer_optimization_20261125_175812/`。首次优化warm1+3rep原始结果、源码及三后端raw另存该目录`optimized_warm1_rep3/`。不是覆盖失败证据后只留新数字。

## 实际优化

- 32线程一个构象；55个重原子的最大分子用线程内步长32覆盖，所有10000构象均真实执行200步。没有删大分子、缩减terms或更换原子图。
- 每个bond/angle/torsion/nonbonded项只算一次E和相关原子梯度，缓存后按原ptr/adj顺序gather回各原子；torsion不再被四个原子重复计算四遍。
- 回溯试探位置只计算各term能量，省去没有用途的梯度运算。原trial energy定义不变。
- terms、ptr/adj与term E/G优先共享缓存，按4/12/24/40KiB动态shared容量分桶；8320构象走全shared缓存。1680困难构象仍完整执行，ptr/adj放shared（最大14752B），TermOut放global scratch（135000320B）；没有64KiB溢出，也不截断1626个term的最难分子。
- 最大梯度gm用并行max归约，max不改浮点加法次序。影响回溯接受的E仍由tid0按原55项以内原子顺序加和，避免并行sum导致离散步长分歧。
- FP64；CPU `-ffp-contract=off`、CUDA `--fmad=false`、HIP `-ffp-contract=off`。没有fastmath或TensorCore替换。

## 协议和真实结果

CPU为一次完整预热+3次完整测量（其算术维持原单线程CPU参考），GPU最终为同一进程/上下文5次完整预热+7次测量；每次从相同未松弛initial坐标独立开始，不在上一次已松弛坐标上继续。GPU输入只上传一次，内核/同步计时不含分配、H2D、D2H和文件I/O；h2d/d2h另列，不能将seconds称全数据准备端到端。

| 后端 | 预热/测量 | 真实raw秒数组 | 中位数秒 | 峰峰幅度/中位数 |
|---|---:|---|---:|---:|
|cpu|1/3|10.560909486, 10.566014739, 10.540641337|10.560909486|0.240%|
|cuda|5/7|0.689139211, 0.690981442, 0.688993549, 0.689565962, 0.689782956, 0.688222897, 0.689281369|0.689281369|0.400%|
|hip|5/7|0.520322000, 0.517412900, 0.526559700, 0.525250900, 0.522464300, 0.522216300, 0.505435800|0.522216300|4.045%|

AMD最终7次并非单调下降，但峰峰仍约4.05%，包含最后较快0.5054秒这一真实点，不删异常值、不再做额外调参，报告中位数和全部样本。GB10约0.40%峰峰，CPU约0.24%。本样本AMD中位数约比GB10快1.32倍，不能推广为通用FP64硬件排名。

旧GPU原始单次为GB10 3.229019秒、AMD3.853299秒，相对新热态中位数约4.68倍/7.38倍；旧样本不是同协议7rep，因此只能标为“对旧单次记录的比值”，不能伪称严格同热态协议的加速比。CPU旧单次10.52215秒与新中位数10.56091秒基本一致。

## 正确性全量验收

- 五份输入binary SHA与旧结果完全一致，epsilon0.5不变。
- 独立NumPy能量五点有限差分18方向均通过，最大绝对误差1.9590129518576305e-10，小于1e-8。
- CPU/CUDA/HIP三端10000构象的有效坐标和完整201步历史彼此逐位完全相同：坐标relative L2=0、完整历史relative L2=0；不仅比较最后均值。
- 新三端输出还分别与各自归档旧原始坐标/完整历史逐位相同，证明性能优化没有改变line-search分支或轨迹。
- 各步能量非增、全有限；同一碰撞定义下三端都是47→0，键长质量不退化；详见优化结果JSON。

## 同源与热态补测审计

最终主仓、AMD仓与远端实际编译source SHA256均为`08f03069a59d7b4717aae34c6a8c2cf9d9f4ec96f4bca86e2fe65d50d3ab7104`。

首次warm1+3后最终热态补测仅在main增加warmups CLI解析，force函数与GPU kernel的全部源文件前缀字节不变，SHA256为`f5bcfc92a11bbeaea917cf06a931d59bca271f802c009ba89ff1bafeb618a861`。两GPU均从同一最终source重新编译并真实跑warm5+7；CPU的力学运算源码未动，因此按父指令保留稳定CPU3rep，不重复40秒测试。

GPU启动前重新查看远端进程，仅见Steam客户端/后台与已有训练器/模拟器服务，未发现游戏本体；保存JSON快照，没有杀任何进程。所有重进程guard，20GiB阈值与250ms统一内存看护。GPU完成已立即通知FX代理启动其最后热态补测。

## 文件与同步

- `amd_hip_port/nvmolkit_batched_conformer/conformer.cpp`：最终同源CPU/CUDA/HIP代码。
- `benchmark_conformer_optimized.py`：自己的验证/基准驱动，默认最终GPU warm5+7，CPUwarm1+3；支持stage cpu/gpu/summary/all。
- `exp12_open_nvmolkit_batched_conformer_hip.py --optimized`：转入该驱动，原stage逻辑保留。
- `results/module14_optimized_results.json`：所有raw计时、完整正确性、覆盖与源码SHA；root module14结果未改。
- `results/module14_optimized_{cpu,cuda,hip}/`：完整原始数组与timings；CPUraw可执行共用GPU字段launch_width，最终汇总已校正CPU实际单线程。
- `logs/module14_optimized_*`与成功HIP构建/运行log保留。

cpp、exp12驱动、优化benchmark驱动和结果JSON已同步独立AMD仓，校验SHA一致；未执行Git命令、未提交、未push。
