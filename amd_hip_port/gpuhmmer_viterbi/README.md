# 原创 CPU / CUDA / HIP Profile-HMM Viterbi

MIT同源C++，56位置、20AA发射log2 odds、M/I/D三状态glocal完整域搜索。输入背景和植入数据库为合成负载，模型发射统计来自真实GB1实验序列；不是官方HMMER/GPU-HMMER全包或校准分数。完整数据与口径见主实验仓 `MODULE12_NOTES.md`。

输入目录：`params.txt`（N stride）、`sequences.bin`（uint8 N×256）、`lengths.bin`（int32 N）、`emissions.bin`（float64 56×20）、`transitions.bin`（float64 MM,MI,MD,IM,II,DM,DD）。输出`scores.bin` float64 N和`timings.json`。所有输入源自同一二进制，三方100k输出逐字节相同。

Linux构建：

```bash
g++ -O3 -ffp-contract=off -std=c++17 viterbi.cpp -o viterbi_cpu
nvcc -x cu -O3 --fmad=false -std=c++17 -arch=sm_121 viterbi.cpp -o viterbi_cuda
```

Windows：将源码/两CMD与input放`C:\AI4S-HIP-Lab\gpuhmmer_viterbi`，执行`build_windows.cmd`后`run_windows.cmd`，现成Strata ROCm SDK只读借用。输入生成与完整跨后端驱动在主仓`exp10_gpuhmmer_viterbi_hip.py`。若只同步源码/JSON，须先在主仓重新生成大输入；Git不附26MiB随机搜索库。

score_one中每线程就地滚动三个double数组；小样本由独立二维表算法验证，全部100k分数压缩存于主仓results。CPU单线程1136ms、GB10234ms、9070XT60.7ms仅为本输入/实现的驻留中位数，传输与冷启动另记，不用于泛化硬件排名。
