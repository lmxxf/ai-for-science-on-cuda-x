# AI for Science 真机工程记录

## 2026-11-23 22:34 +08:00 — 启动

- 已完整读取 `EXPERIMENT_PLAN.md`。分为 RDKit、Uni-Mol、CUDA-X 三条实施线，最后统一串联与校验。
- 硬件实际探测：NVIDIA GB10 / DGX Spark，驱动 580.142，CUDA 13.0，基础 PyTorch 2.14.1+cu130；GPU 初始空闲。
- `.venv` 使用 Python 3.12、system site packages，并以 `.pth` 复用 `/home/lmxxf/work/interp-curvature-exp/venv/lib/python3.12/site-packages`。安装真实 aarch64 RDKit 2026.9.2 wheel。基础 torch、sklearn、pandas、cairosvg 已导入成功，CUDA 可用。
- 添加 `guard_process.py`：每 0.25 秒检查 MemAvailable；低于 20 GiB 时终止整个子进程组，写入 JSON 审计。训练保持低 batch / workers。
- 已确认 WenQuanYi Micro Hei 字体可用。所有图以真实结果生成，CPU/GPU瓶颈与模型胜负不预设。
- 图纸待实测校正：TPSA 不是血脑屏障的硬阈值；二代抗组胺药的中枢作用还受外排转运体等影响。不会把分子量/LogP/TPSA当作临床嗜睡判定器。

## 2026-11-23 22:36–22:41 +08:00 — 三模块真机运行

### RDKit

- 下载真实 `delaney-processed.csv`（1128行）与 `BBBP.csv`（2050行，2039合法，11非法）。原CSV及SHA256保留。
- 命令：`.venv/bin/python exp1_rdkit_data_processing.py --stage prepare`，随后运行 `--stage benchmark`；1000分子×1/10构象，单核/8进程四格均真实跑完。
- 首次计时：1构象单核7.927s/8进程1.385s；10构象单核67.209s/8进程9.614s。1128训练缓存独立生成约8.870s。MMFF200步未收敛不能冒充失败或已收敛，分别统计。
- 发现手写非索非那定结构多两个碳（MW529.72），改用PubChem CID3348官方结构（MW501.67）。七种药逐一核对PubChem分子式，原始响应保留 `data/pubchem/`。
- 图1/图2从真实结构与坐标生成；后续将图2换为可命名且MMFF已收敛的硝苯地平，并在最终结果固定原子编号与距离。

### Uni-Mol

- 从官方源码安装 `pip install --no-deps -e vendor/Uni-Mol/unimol_tools`，兼容Python3.12/aarch64，避免旧numpy/pandas pins触发重建旧包。权重为官方181MiB no_h预训练模型；真实15层/512维/64头/47,331,139参数。
- 命令：`.venv/bin/python guard_process.py --audit logs/module2_memory.json -- .venv/bin/python -u exp2_unimol_finetune.py --epochs 20 --batch-size 32`，完整stdout留在 `results/module2_train.log`。
- ESOL按canonical母核去重1117，固定seed42划分893/112/112，目标归一化只拟合train，validation选checkpoint，test最后一次评估。
- 20轮全参数训练+逐轮验证37.183s，第13轮最佳。测试：均值RMSE2.175708，Morgan RF1.144189，Uni-Mol0.573779/R²0.930432。不能把预训练/容量/3D一同变化的收益全部归因3D。
- GPU看护最低可用107.197GiB，峰torch分配2.234GiB，GPU已释放。CPU单构象8.870s比20轮训练更短，原图纸“CPU必为最大瓶颈”的叙事不成立。

### cuSpa / cuPotential

- CMake+nvcc13从原版cuspa编译sm_121成功；配置、编译日志保留在 `results/cuspa_build_*.log`。Numba/NumPy版本问题通过 `results/module3_builddeps` 独立路径解决，没有改基础venv。
- 命令：`python guard_process.py --audit logs/module3_memory.json -- python exp3_cudax_cuspa_benchmark.py`，原版CUDA核使用Torch提供设备buffer。高层聚合要求CuPy，实际直调官方 `_core.aggregate_to_cells`。
- 首次float32完整校验16点不同；未删点，双方改float64、同种子全量重测，200万点归属和50k×500 CSR均零差异。float32具体逐点原因尚未定位，不宣称完全修复。
- CPU为单线程编译后同类网格+射线法；完整设备驻留CPU272.274ms vs GPU9.636ms（28.255×），仅查询156.105×。输入传输14.502ms另列；相加估算11.280×，不含输出回传/JIT/编译。原始多次计时数组保留。
- cuSpa 11个源头文件2378行；atomicAdd用于网格计数，基因聚合实为CUB排序+游程编码。未检出TensorCore/内联PTX，但HIP/hipCUB/绑定仍需移植，AMD未实测。
- cuPotential 62个C++/CUDA源头文件16650行；公开融合积分/约束/GroupedBonded有实质内容，核心libcupo/API/LTOIR属专有许可。clone无libcupo.so，构建从release获取。只作源码审计，未伪称运行其MD。

## 2026-11-23 22:41 +08:00 — 串联与可视化核对

- 添加 `run_pipeline.py` 顺序入口与 `build_report.py` 从实测JSON汇编报告。`--reuse-results` 明确只核对现存结果，manifest记为validated_cached_result，不能冒称新实验。
- `skill/ai4s-mol-skill/SKILL.md` 已通过skill-creator的quick_validate。
- 四张640px白底SVG均CairoSVG转PNG，主线程逐张打开目检，未见中文缺字或重叠。
- 分子构象是模型候选几何，cuSpa数据是明确标注的合成负载；报告区分事实、组合估算、尚未完成的AMD移植和BBBP分类训练。

## 2026-11-23 22:43 +08:00 — 最终交付校验

- RDKit默认完整入口额外实跑通过，最终四格墙钟为7.961s、1.146s、66.608s、9.987s；首次计时保留在早期日志。完整重跑后仍保留1128行缓存，未被1000行测速覆盖。
- 最终图2为硝苯地平，原始行48，25重原子；原子1/24（均O）相隔7根键，3.4527025Å，所选构象MMFF已收敛。
- `verify_artifacts.py`独立重算测试RMSE/R²，检查train/val/test不交叉、去重结构数、validation最优轮次、1128完整构象缓存、坐标距离、CPU/GPU零差异、药物公式、原CSV哈希、四SVG白底/字体/PNG、内存看护，全部PASS。
- 一键入口在完整三模块后会自动进行上述校验并汇编REPORT；本轮各实验模块已独立真机执行，最终串联通过`--reuse-results`验证现有产物，未冒称重新训练。manifest记录准确复用模式与哈希。
- `REPORT.md`依据最终JSON生成。cuPotential仅源码审计、BBBP未训练分类、AMD未移植、float32差异未根因定位均已明确；ESOL训练与cuSpa float64正确性验证完成。

## 2026-11-23 23:28–23:33 +08:00 — cuSpa 移植 AMD RX 9070 XT

- 完整读取 `AMD_9070XT_PORT_PLAN.md`，`ssh amd9070 hostname`连接成功。通过Windows Win32_Process检查全进程：有Steam、修改器、MuMu等后台，无游戏本体；没有关闭任何用户进程。
- 工作目录固定 `C:\AI4S-HIP-Lab`，复用Strata现成`.rocm-win`的ROCm10.2 HIP工具链及VS vcvars64，不写Strata原工程。长PowerShell命令通过UTF-16LE EncodedCommand传输，避免Windows cmd对管道与正则引号的二次解析；初次裸引号命令失败后已改。
- `amd_hip_port/hipify_headers.py`机械翻译上游5个算法头（CUDA runtime→HIP、CUB→hipCUB）。原几何、索引、64位键排序/RLE/CSR逻辑保留；独立C++ harness绕过nanobind。原Apache2许可、逐行diff、源/目标SHA256均保留。
- `export_module5_reference.py`在Spark上按原种子20261123生成完全相同的50k八边形/2m点/500基因float64输入，导出46MiB裸二进制与header哈希、精确网格参数。原版GB10库重新生成完整assignment/CSR参考，匹配793204点、NNZ780563；输入scp到远端，不依赖两机自行生成随机数。
- AMD编译：`amd_hip_port/build_windows.cmd`，clang++ `-x hip --offload-arch=gfx1201 -O3 -ffp-contract=off`，hipCUB/rocPRIM，编译一次成功。远端执行 `cuspa_hip.exe ..\data ..\out_amd 7`；日志在`amd_hip_port/build_windows.log`、`run_windows.log`。3次预热，7次计时；无fast math。
- AMD同步墙钟中位数：index2.3869ms、query0.9207ms、aggregate3.7683ms、完整驻留6.0502ms；H2D单次11.6143ms，D2H单次1.5167ms；冷启动34.1987ms。分开阶段中位数不要求相加等于完整流程。
- 为对齐入口，本地同源harness也用`nvcc -x cu -std=c++17 -O3 --fmad=false -arch=sm_121`编译原CUDA头，在统一内存看护下运行：GB10 index1.658594ms、query1.519842ms、aggregate5.563753ms、驻留5.759897ms，H2D0.878337ms、D2H0.297073ms。原Python9.636ms是另一入口，不能拿它与AMD6.050ms宣称硬件胜负。
- 主线程独立运行`compare_module5_outputs.py`：所有2000000归属、50001行指针、780563列索引、780563计数逐值与逐字节相同；四个SHA256全一致。GB10同源C++也与原版库输出逐字节一致。仅声称最终整数输出一致，不声称所有浮点中间态一致。
- 更新`results/module5_amd9070_results.json`与`results/module5_comparison.json`，更新图4（三方条形、同源入口、AMD阶段/传输、全量哈希对账），CairoSVG重渲染PNG并目检。`build_report.py`已纳入模块五，重生成REPORT不会丢AMD结果。
- 主目标已完成，不追加图纸可选分子势能微核；未声称移植cuPotential专有运行时或完整Python包。AMD实验进程已结束。

## 2026-11-23 23:41 +08:00 — 第二阶段启动

- 完整读取`PHASE2_FULL_AI4S_PLAN.md`；用户授权书房RX9070XT独占使用，现有Strata文件仍保持只读、AMD实验写入`C:\AI4S-HIP-Lab`独立子目录。
- 按模块六→七→八执行GPU实验；代码、数据下载与CPU校验可预先并行，避免共享GPU争用污染计时。
- 模块六：真实BBBP分类、validation挑checkpoint/test一次、模型实算原子遮挡归因、真实权重Gaussian/pair-bias/attention CUDA-HIP核。
- 模块七：原创开源PBC/27邻格/LB混合LJ+reaction-field Coulomb与解析力；小体系暴力对照与有限差分先验，再测32k/128k，不声称复刻专有libcupo全部功能。
- 模块八：真实蛋白实验序列、Biopython、ESM-2真微调、按模型输出做突变扫描，不能将模型预测热图当成新的实验测量。
- 仓库初始状态：主仓仅新图纸未跟踪，HEAD00fb2d4；AMD独立仓干净，HEADc90e8f3。用户要求最终两个工作区干净，完成后将本次源码/文档/轻量结果同步并本地提交，重权重/缓存/编译产物留本机不入版本库。

## 2026-11-23 23:43–23:49 +08:00 — 第二阶段真机结果与独立核验

### 模块六

- `.venv/bin/python guard_process.py --audit logs/module6_memory.json -- .venv/bin/python -u exp4_bbbp_unimol_and_amd.py --stage train`：20轮全参数BBBP训练112.259s，CPU8进程构象12.447s。11冲突母核22行整组剔除、同标签重复62行移除；1955候选构象成功1952，1058在200步MMFF收敛，894保留最后坐标。
- 固定分层划分1561/195/196，validation ROC选第4轮。Uni-Mol test ROC0.933333/AP0.975010/Accuracy0.892857；RF ROC0.932609/AP0.975935/Accuracy0.903061，基本打平，不预设3D胜出。
- 苯海拉明/非索非那定逐重原子遮挡已实算（概率0.9573/0.03849）；两种药都在训练集，不能当独立泛化证据。删除token和距离行列、保留其他坐标，不重优化，归因仅是敏感性。
- guard运行`amd_hip_port/unimol_pairbias/run_pairbias.py`导出真实checkpoint参数与LayerNorm后输入，顺序编译运行CUDA与HIP，完整默认all入口已串联。Gaussian128→128/GELU→64、首层QKV、masked softmax、AV；8药/39pad/FP32。GB105.238ms、AMD12.101ms；跨GPU及对官方PyTorch max_abs1.526e-5。朴素核不是成熟BLAS或全模型AMD。

### 模块七

- 原创MIT C++：CPU g++、CUDA nvcc sm121、HIP clang gfx1201，独立目录cupotential_open。CPU准备与NumPy全配对/有限差分先执行，待模块六GPU释放后顺序GPU。
- PBC/MIC/27邻格，rc10Å/介电78.3、LB LJ+reaction-field Coulomb、解析F=-gradE；合成中性电荷晶格、无键拓扑/排除规则，LJ未shift，非PME、非完整MD。
- 216原子暴力参考受力相对L2=3.127e-16，有限差分六方向max_abs1.052e-10，净力2.82e-15，周期平移相对L2=1.23e-15。
- 32k CPU61.168/CUDA7.993/HIP7.901ms；128k CPU242.567/CUDA27.188/HIP22.426ms。全部原子能量/受力数组逐值完全相同，主线程额外检查非零力范围与SHA一致。总E C++ long double跨OS末位可异，保留reported与统一NumPy归约canonical E，不将归约末位差误称核误差。

### 模块八

- 模块七GPU全部结束后启动guard包裹的`exp6_protein_esm2_finetune.py --stage all`，实际12轮/456优化步，15.016s，val选第11轮。
- 真实FLIP GB1 sampled8733唯一标准AA序列；固定seed20261124抽3000，2400/300/300；变异39/40/41/54，保留完整前56AA GB1域、裁掉全库恒定融合尾巴。Biopython五描述符+AAC/二肽共425维，特征0.222s。
- 官方ESM-2实际7512122可训练参数/6层320维20头/FFN1280/33词表/RoPE；33词表区分X/B/U/Z/O等残基与特殊token。第一层query权重相对预训练L2变化1.657597，确认并非只训练回归头。
- 同测试集RMSE常数1.243687/RF0.870472/Ridge0.931356/ESM0.538136，ESM Spearman0.898290/R²0.812710；仅val选alpha和checkpoint，标准化train-only，test最终各模型一次。
- 80格单点突变预测扫描完成；四个WT自替换delta=0，不称新实验验证。

### 总校验

- `verify_phase2.py`独立重算BBBP ROC/AP、GB1 RMSE/R²/Spearman，核对split互斥、最佳轮次、归因原始差值、真实force/energy数组、突变差值/序列/哈希、三SVG白底字体/禁ᵀ及PNG，全部PASS。
- 图5/6/7均640px白底中文，主线程逐图view_image目检无重叠缺字。所有GPU已释放。
- `run_pipeline.py --phase 2 --reuse-results`通过，明确只串联已执行产物而不冒称重复训练；真实单模块执行日志全部保留。`build_report.py`会自动汇编第二阶段，Skill及README同步更新。

## 2026-11-23 23:50 +08:00 — 双仓同步与交付

- `sync_phase2_to_amd.py`只同步源码、许可、脚本、图5/6/7及紧凑JSON/日志；大输入张量、训练权重与原始力数组留原实验工作目录。AMD仓`verify_results.py`核对同步文件SHA与轻量数值证据；主仓`verify_phase2.py`已独立核对完整原始数组，两者不混称。
- AMD新增源码统一放`amd_hip_port/unimol_pairbias/`与`amd_hip_port/cupotential_open/`。移除本次子任务造成且内容完全相同的未跟踪重复顶层cupotential_open副本，保留唯一规范路径。
- 所有实验已结束，GPU释放；完成最后报告与同步哈希检查后本地提交两仓。本轮未执行git push或对外发布。
- 最后回归发现第一阶段旧校验器把目录SVG总数写死为4；新增图后误报。已改为明确核对fig1–fig4，第二阶段继续独立核对fig5–fig7；全阶段入口`--phase all --reuse-results`随后PASS。
- 原始Windows编译/计时日志保留CRLF和工具输出尾空格，不为Git空白检查改写实验原始日志；源码差异单独检查。
