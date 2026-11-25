# 非键核空间重排优化：最终保留 v0 默认，候选与回退完整记录

2026-11-25。源码`amd_hip_port/cupotential_open/open_nonbonded.cpp`；独立驱动`benchmark_nonbonded_optimized.py`。没有修改主`module7_results.json`、README、REPORT、DevHistory或文章；父进程负责发布账本。

## 固定任务与数据

同原32k/128k合成带电原子体系，周期盒、27邻格、10Å截断、LJ/RF公式、参数和位置全未变：

- 32k atoms.bin SHA256 `003cec4c76d285d8fbc377f6ebb0de43be5e6c41269d59dcaa54174686ad2a32`；
- 128k SHA256 `74e421bf4e3929aa8e6f4e2e54ef23a65491f9cc06e8bb7050c08bde8cccfa80`。

本地逐个核对原metadata，远端在执行前对同两个atoms.bin做完整SHA检查。所有双精度，CUDA禁FMA/HIP禁contract，无fastmath。GPU配置统一128threads，邻格访问严格dz→dy→dx，格内稳定原ID顺序保持。

## 变体及空间布局

- **v0默认控制**：原核，稳定分桶/排序索引；原位置线程按sorted ID间接查原AoS6原子；每对`epsilon=sqrt(epsilon_i*epsilon_j)`。
- **v1精确AoS重排**：sorted后把原子复制为连续AoS7，sorted位置线程求力，按`sorted[pos]`写回原ID；邻居按sorted槽位连续查。每对仍精确sqrt乘积，不变算术次序。
- **v2预sqrt AoS候选**：相同AoS7，只将epsilon混合改为预存sqrt(epsilon_i)再乘sqrt(epsilon_j)。这是代数相同、FP64舍入可不同的单独候选，未冒称与原数组逐位相同。
- **v3最后SoA候选**：明确的7个field-plane，每plane连续N个double；中心与邻居字段按标量读取，sorted位置线程求力、原ID写回。epsilon分解、27邻格和格内顺序完全与v2相同。只加了这一次SoA，不再调tile或硬追性能胜出。

v0–v2物理函数前缀在加v3前后逐字节不变，前缀长度/哈希记录`results/module7_round1_kernel_prefix.json`，SHA256 `70029cd89542aa7a0f3214c58f0cff2e6e9826bd4ee276aace341e88bfc984be`。

原CLI仍进入legacy v0；新CLI为 `EXE INPUT_DIR OUTPUT_DIR opt 7 5`（allvariants），追加`03`只跑新source的v0/v3控制对照。**推荐仍保留v0默认；v3是测试候选，不叫优化全面成功。**

## Round1：v0 / v1 / v2，同一harness5warm7

下列毫秒为完整index＋reorder＋force的独立中位数，重排成本计入；阶段明细与7个原始值见JSON。

| 原子数 | 后端 | v0原核 | v1精确AoS | v2预sqrt AoS |
|---:|---|---:|---:|---:|
|32k|CPU单线程C++|61.091233|56.607225|55.748629|
|32k|GB10|7.992634|9.298625|8.858159|
|32k|RX9070XT|7.723200|7.422200|6.256000|
|128k|CPU单线程C++|240.924563|229.247413|227.634124|
|128k|GB10|27.115007|32.811133|31.318372|
|128k|RX9070XT|21.077100|21.284300|19.849600|

AoS重排让CPU与AMD部分任务变快，但GB10回退；预sqrt比精确重排版少算sqrt，但不能据此称原核在两卡都被超越。Round1全部快慢结果都保留在`results/module7_optimized_round1_results.json`、`results/module7_optimized/{cpu,cuda,hip}/{N}/v0,v1,v2/`。

## 唯一追加v3：新source两卡v0/v3同harness对照

5次完整预热、7次完整测量；CPU性能复用本轮稳定v0参考，未给未测的全CPU v3贴上旧v2耗时。

| 原子数 | 后端 | fresh v0完整ms | v3完整ms | v3最小–最大ms |
|---:|---|---:|---:|---:|
|32k|GB10|7.991593|8.781069|8.779549–8.801134|
|32k|RX9070XT|7.767300|7.111100|6.800600–7.539000|
|128k|GB10|27.137404|31.023216|31.014975–31.044880|
|128k|RX9070XT|20.947800|20.222700|20.073100–23.257700|

结果仍混合：GB10 v3比新v0多约9.9%/14.3%耗时；AMD少约8.4%/3.5%，128k收益小于原始时序波动区间，不能称稳健收益。不做GPU赢家拼接（如GB10选v0、AMD选旧v2）。发布默认共同v0，v2/v3分别列实验记录。

v3分账中位数（ms）：

| 原子数/后端 | index | reorder | force | 完整独立计时 |
|---|---:|---:|---:|---:|
|32k GB10|.073344|.010160|8.704893|8.781069|
|32k AMD|.147700|.079800|7.056500|7.111100|
|128k GB10|.087232|.036480|30.904095|31.023216|
|128k AMD|.212900|.090300|20.026400|20.222700|

阶段和完整流程各自独立同步计时，阶段中位数不能直接相加当完整中位数；AMD时钟/缓存/系统波动均实录。v0 reorder阶段无重排kernel，微小计时仅同步/计时器开销。优化原因未做硬件计数器剖析，不把缓存线猜测写成已证实根因。

## 完整数值与独立物理验算

不是只验总能量：每个规模/变体核对所有N个energy数组元素与3N个force元素，报告向量relativeL2、maxabs与逐位相等标志。

- v1对CPU v0完整E/F逐位相同；
- v2、v3对原公式CPU v0的E/F relativeL2约6.5e-17，全部<1e-15；energy maxabs3.55e-15、force maxabs1.78e-15，阈值明确是向量相对L2，不偷换为绝对最大误差；
- Round1同variant CPU/CUDA/HIP输出彼此逐位相同；v3 CUDA/HIP互相逐位同，并与Round1 GPU v2输出逐位同；未声称测过全规模CPU v3时间；
- v0新source结果与原公式CPU参考逐位同，旧raw始终保留。

216原子独立NumPy O(N²)物理检查重新执行：v3 force relativeL2=3.19048e-16、总E relative误差1.58354e-16；净力L2=4.07204e-15；周期平移force relativeL2=1.23240e-15。独立势能中心有限差分6分量最大绝对误差1.05155e-10。旧v0/v1/v2的独立检查也保留，未拿总E相同代替逐原子检查。

物理原型仍是LJ＋反应场，不是PME或完整MD；该范围与旧模块完全一致。没有改粒子分布、截断、电荷、epsilon或位置来使优化好看。

## 证据、复现与安全

- 最终`results/module7_optimized_results.json`：`default_variant/common_variant=v0`、`candidate_variant=v3`，每规模每后端默认`selected_timing`为v0，候选另给`candidate_timing`，CPU明确引用真实v0时间；
- Round1 `results/module7_optimized_round1_results.json`；新v0/v3 raw `results/module7_optimized_soa/{cuda,hip}/{32000,128000}/`；
- 独立物理结果`results/module7_optimized_physics.json`及`module7_optimized_soa_physics.json`；
- 原CPP/exp5驱动/JSON/原始两规模三端输出归档`results/module7_optimized_previous/`。大raw留本机，不作为Git blobs提交。

驱动已修复干净build目录缺exe的问题：`benchmark_nonbonded_optimized.py --stage all/local`先guard编译CPU/CUDA的optimized可执行。`all/local`会检查固定输入，缺失时用原seed/公式生成并强制核对原SHA，先guard编译CPU/CUDA，再生成缺失CPUv0参考与本次03物理/数值证据；远端缺输入会上传相同原字节再验SHA。`all`跑本地和远端、`remote`只续远端、`summarize`只读汇总。Round1 JSON/v2 raw是可选历史，缺失明确Unavailable，不阻碍本次完整CPU/crossGPU/独立物理硬验收。`exp5_open_cupotential_hip.py --optimized --stage all`只转发新驱动，exp5不带flag仍走旧v0。复现P1已以空目录mock和实际无历史raw汇总测试通过，未再执行GPU。

所有Spark重进程经250ms/20GiB guard。远端借用现有SDK/VS，只在独立cupotential_open子目录构建。先读取Win32_Process全名单、核验原输入SHA，未发现游戏本体，保留Steam/trainer/MuMu，未杀用户进程。最终源码与独立驱动同步两仓和远端，并有SHA证据。发布git交给父进程，本任务不commit/push。
