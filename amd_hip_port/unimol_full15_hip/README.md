# Uni-Mol v1 全15层：原生 HIP / hipBLAS

真实ESOL checkpoint的47,331,139参数与原112个留出分子，全部15层、pair传播、LayerNorm、FFN、真实线性回归头端到端在RX9070XT执行。源码`unimol_full.cpp`，原型遵循上游MIT许可（附DP Technology原许可）；权重仍遵循其来源许可，不由此改变。

## 验证

- 全FP64：HIP对同公式GB10 FP64最大预测差7.105e-15 logS，RMSE0.573633556/R²0.930466813；共50,501,632个逐层node/pair元素核验，最大约9.1e-13。
- 官方FP32兼容模式：仅Gaussian完整运算使用官方FP32路径，其余15层为FP64。HIP对官方完整FP32最大差2.268e-6<1e-5；RMSE0.573633402。源码参数开关`gaussian_fp32`，不是预测后处理校准。
- 原BF16 autocast预测/指标完整保留，不能要求不同精度硬贴原RMSE0.57377856。全双精度对官方FP32一度有1.92e-5差，已通过定位Gaussian helper强制float路径补上兼容模式，不隐藏原结果。

真实网络：embedding LN、Gaussian128/128GELU/64、15轮pre-LN QKV/scaled attention/pair累加/softmaxAV/out_proj残差、FFN512/2048GELU/512残差、encoder末尾LN、LinearHead512→1。不存在图纸假定的两层tanh头。BOS/EOS保留，单分子无padding；不在AMD训练、不重新选test。

## 复现

主实验仓：`.venv/bin/python exp11_unimol_full15layer_amd9070.py`导出真实权重和输入、在GB10算独立参考，再用SSH在`C:\AI4S-HIP-Lab\unimol_full15_hip`编译/运行本程序。默认两精度路径均执行；`--stage mixed`只运行官方FP32兼容补验。

远端`build_windows.cmd`复用现成Strata ROCm10.2 SDK与vcvars64，只读Strata，链接hipBLAS；`run_windows.cmd`执行全FP64。输入是195张量清单和FP64权重、112分子的tokens/edge/distance，不包含预测或标签；输出预测与全逐层trace。权重379MB、trace数百MB留实验机器，不随Git复制，须从主仓checkpoint重新导出。

`results/module13_results.json`包括全部预测、真实目标、旧BF16保留、逐层误差、设备事件与墙钟两种时间。全FP64累计设备GB101.756s/AMD2.291s；兼容模式AMD5.406s另记。均为精度优先逐分子参考，未经batch生产优化，不作硬件总性能排名。
