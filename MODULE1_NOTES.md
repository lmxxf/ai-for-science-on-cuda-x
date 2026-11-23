# 模块一：RDKit 真机记录

实验日期：2026-11-23。执行环境：DGX Spark GB10，RDKit 2026.09.2；固定随机种子20261123，ETKDGv3 `numThreads=1`，MMFF94最多200迭代；多进程版本固定8 worker。

执行：`.venv/bin/python exp1_rdkit_data_processing.py`。可分别使用`--stage prepare|benchmark|figures`。正式一键完整重跑的 stdout 在`results/module1_full_run.log`，最终权威汇总是`results/module1_results.json`。

## 数据与清洗

- ESOL：[DeepChem MoleculeNet原始CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv)，1128行，全部通过RDKit解析。
- BBBP：[DeepChem MoleculeNet原始CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv)，原始2050行，2039合法，11非法，不应将2050与2039混用。
- SHA256与真实清洗/描述符耗时记录于`results/rdkit_preparation.json`，清洗数据为`data/esol_clean.csv`、`data/bbbp_clean.csv`。保留`source_index`以回查原记录。
- `SaltRemover`先去掉已知盐碎片，再保留最大重原子连通片；不做标签重测、pH指定的电离标准化或互变异构标准化。母体与完整盐的测量性质可能有差异，因此清洗是明确的实验建模选择，不是唯一正确的药物标准化流程。
- `C(C)(C)(C)(C)C`为人为构造的非法五价碳拒绝示范；`.Cl`样本为人为加盐的真实苯海拉明结构示范，不冒充来自ESOL的脏记录。该写法代表中性母体与HCl碎片，不能误称精确建模了溶液质子化状态。
- 七种具象药物的分子式均已逐一对账PubChem官方API，原始响应保存在`data/pubchem/*.json`，核对表`results/drug_structure_audit.json`。非索非那定最初手写结构有误，多出了碳链，已改成PubChem结构后重跑，最终C32H39NO4、MW501.667。

## 构象与计时

计时四格均真实执行1000个分子；使用ESOL前1000行而非按大小筛选。每格都重新嵌入和优化，不把前一次缓存当作新计算。1/10构象均固定种子，8进程墙钟包括进程建立与IPC。具体秒数和收敛计数见`results/rdkit_timings.json`；操作系统负载会影响秒数，不预设CPU为整条流水线瓶颈。

全1128行单构象缓存为`results/esol_conformers.pkl`，字段包含`source_index,smiles,target,atoms,coordinates,nconfs,mmff_converged,mmff_supported,seconds`。原子为重原子、坐标为float32 Å，按清洗CSV行序排列。首先完成单核1000行计时，再单独补128行。总计时在`results/rdkit_full_conformer_time.json`；它包含剩余128行计算及小量缓存写入开销。所有1128行都嵌入成功，但只有997行在200次MMFF迭代内收敛，131行仍保留最后构象供后续模型使用；生成成功与力场已收敛不同。

## 可视化与真实折叠案例

- `visuals/fig1_rdkit_cleaning_and_bbb_drugs.svg`：去盐前后、苯海拉明与非索非那定的实算描述符与2D结构。所有氧原子高亮，不将高亮等价为独立因果证明。
- 苯海拉明：MW255.361 / LogP3.3542 / TPSA12.47 Å²；非索非那定：MW501.667 / LogP5.5105 / TPSA81 Å²。这里LogP是中性结构的RDKit Crippen值，不能直接当生理pH下logD。更高LogP并不保证更强穿脑。
- 低脑暴露还受电离、外排运输及剂量影响。TPSA90不是硬阈值，第二代也不等于任何人都绝不嗜睡。参考原始研究：[PMID17180728](https://pubmed.ncbi.nlm.nih.gov/17180728/)、[PMID12584158](https://pubmed.ncbi.nlm.nih.gov/12584158/)。图为机制教学，不是用药建议。
- `visuals/fig2_1d_smiles_to_3d_conformer_distance.svg`：硝苯地平nifedipine，ESOL原始行48，25重原子；MMFF已收敛的首个构象中，重原子1与24均为O，拓扑最短路7根键，空间距离3.4527025 Å。候选筛选还要求路径至少4条非环单键，并排除MMFF未收敛记录；使用可命名常见药而非全库最短接触。
- `results/folded_molecule.json`保存完整XYZ与原子索引（从0开始）；`results/folded_molecule_distance.npz`保存25×25欧氏距离矩阵（对称误差0、对角0）。这是计算构象，非晶体学实验实测，不能说此距离在所有环境固定。
- 两图均640px宽、白底、WenQuanYi Micro Hei，已用cairosvg转PNG并目检，中文正常、没有元素越界或文字遮挡。

## 边界

本模块生成ESOL全量3D缓存并交由模块二复用；BBBP完成真实清洗与描述符，不重复做未使用的全量3D。ETKDG生成的是一个可能构象而非蛋白结合态、溶液构象分布或量子化学基态。10构象计时记录全部生成与优化，不按标签选择构象，不保证10个构象彼此去重。
