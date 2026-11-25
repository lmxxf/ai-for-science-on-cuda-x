#!/usr/bin/env python3
from pathlib import Path
import json
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
def report_text():
 if not all((R/f'results/module{n}_results.json').exists() for n in [14,15,16]):return ''
 c,d,a=[load(f'results/module{n}_results.json') for n in [14,15,16]];p=c['preparation'];g=c['gradient_validation'];cpu=c['backends']['cpu'];failure=c['failure_case']
 text='''

# 第四阶段：批量构象松弛、全GPU驻留本征求解与381终审

## 模块十四：10,000构象×200步的原创开源批量优化

ESOL清洗表前1000条记录（991个不同canonical母体；吞吐按输入记录计数，不另作去重），每条生成10个全新ETKDGv3构象，**没有拿已经MMFF优化的坐标当起点**。显式H参与ETKDG后导出重原子，再加入固定种子0.06Å高斯扰动，留下完全可复现的新初态。不是1000种均为临床药物，ESOL还包含一般有机物等。

'''
 text+=f"8进程ETKDG、扰动、拓扑和参数准备墙钟{p['preparation_wall_seconds']:.6f}s；worker累计{p['etkdg_plus_jitter_sum_worker_seconds']:.3f}s不是串行墙钟。最多{p['max_heavy_atoms']}重原子，10k有效原子总数{p['heavy_atoms_total_all_conformers']:,}。\n\n"
 text+='力场为明确的简化重原子模型：\n\n- 谐振键k400，平衡键长从MMFF读取（可用时）；cos-angle项k40、平衡角从MMFF读；扭转项k0.2、周期2/3。\n- 拓扑间隔>=3根键的非键对，softcore LJ12-6＋库仑，Gasteiger电荷将H贡献折入重原子；最终epsilon0.5kJ/mol、sigma=0.85×两原子vdW半径之和。\n- MMFF API在此只读部分平衡几何参数，**没有运行MMFF优化器，不是完整MMFF94或真实nvMolKit替代品**。能量kJ/mol、距离Å，所有参数和排除规则可查看源码/README。\n\n'
 text+='| 每1000条拓扑记录 | 成键 | 角 | 扭转 | 非键对 |\n|---|---:|---:|---:|---:|\n'
 t=p['topology_terms'];text+=f"| 项数 | {t['bond']:,} | {t['angle']:,} | {t['torsion']:,} | {t['nonbonded']:,} |\n\n"
 text+='''一个block处理一个构象，坐标/候选/梯度状态驻留shared，确定性求和与回溯下降在kernel内连续200轮。拓扑参数仍从global读取，能量诊断也写global，准确说法是“**坐标迭代无需CPU往返**”，不是整个kernel零global访存。单原子无力场项的10个构象自然保持不动，不编造下降。200轮不等于所有构象达到严谨全局极小。

'''
 current_conformer=c.get('current_steady_state')
 text+='| 同源同任务：10k×200 | 优化后稳态中位秒 | 初始均能量 | 最终均能量 | 近碰撞对数：前→后 |\n|---|---:|---:|---:|---|\n'
 for bk,old_backend in c['backends'].items():
  b=current_conformer['backends'][bk] if current_conformer else old_backend
  initial=b.get('mean_initial_energy',old_backend['energy_initial_mean']);final=b.get('mean_final_energy',old_backend['energy_final_mean'])
  before=b.get('collision_pairs_initial',old_backend['nonbonded_collision_pairs_initial']);after=b.get('collision_pairs_final',old_backend['nonbonded_collision_pairs_final'])
  text+=f"| {bk} | {b['seconds']:.6f} | {initial:.6f} | {final:.6f} | {before}→{after} |\n"
 if current_conformer:text+='\n2026-11-25优化后GPU预热5次＋7次，CPU1次＋3次；每次从相同初态重置。旧单次CPU10.522s/GB103.229s/AMD3.853s保存在原backends，新时账见current_steady_state。8320个构象共享全缓存，1680个完整fallback，未删样本。\n'

 text+=f'''
全部10,000构象能量不升，9990严格下降；平均键长绝对偏差{cpu['bond_abs_deviation_initial_mean_A']:.6f}→{cpu['bond_abs_deviation_final_mean_A']:.6f}Å。三后端全部400,350个有效坐标、10,000个最终能量及10,000×201整段历史逐位一致，L2=0。不是只比较平均数。

独立NumPy能量与C++解析梯度，18方向五点差分（h=2e-4Å）最大误差{g['max_abs_error']:.3g}<1e-8。碰撞固定定义：图距离>=3且空间r<0.8sigma，阈值前后没改。

### 保留一次失败：能量下降、跨卡一致仍可物理不合理

初版epsilon={failure['original_epsilon_kj_mol']}时，优化能量虽下降，近碰撞却{failure['original_contact_initial']}→{failure['original_contact_final']}，不能称构象质量通过。失败三平台raw历史/坐标、JSON和旧参数都留存。只提高明确排斥强度到{failure['revised_epsilon_kj_mol']}，在完全相同初始坐标上复验，碰撞47→0；不同参数的绝对能量不可当作同一势能函数横比。

初始SHA256 `{failure['initial_coordinate_sha256_before']}`前后不变，旧/新terms哈希分开记录；没有换起点、删失败样本或移动阈值。这里只证明这组简化模型和输入的数值/几何检查通过，不宣称参数具备生产级化学验证。

### 旧RDKit66.608秒的比较边界

原单核66.608s/8进程9.987s包含ETKDG＋显式H的**真正MMFF94优化**；新任务只计另一简化重原子力场的200轮，输入准备又另计3.534s。两者不能直接相除宣称“MMFF94加速20倍”，也不能从这次实验宣称已经GPU实现ETKDG或全部nvMolKit。公平速度比较是上表的同源CPU C++/CUDA/HIP；本次为单次完整任务计时，不伪称重复稳态中位数。

![10000构象松弛](visuals/fig13_open_nvmolkit_10000_conformers_amd_vs_nvidia.svg)

## 模块十五：Jacobi和MGS也进GPU，全驻留Block-Davidson

每个k点由一个持久block负责，32×32共享内存循环Jacobi、两遍MGS、H·V、投影、Ritz重构、残差/对角预条件和收敛控制都在设备内执行。每轮无标量/波函数主机往返；主机只最初上传H、最后下载特征对与历史。H/基向量等工作态仍用设备global显存，不把驻留等同于全部shared。

第一版32维每步丢掉额外Ritz方向，在近简并根上停滞，未达精度的pilot记录保留，没有当成功。改为厚重启：保留24个Ritz向量（16目标＋8护根），再补8个最大目标残差方向；仍32维子空间，真正实现于GPU，无CPU答案回填。最终64/256系统均15–16轮、device残差门槛1e-10。

| 系统数 | 路径 | 含首尾分配/传输总秒 | 驻留核秒 | 对精确解max_abs | 完整H残差 |
|---:|---|---:|---:|---:|---:|
'''
 for s in d['systems']:
  for bk,b in s['backends'].items():
   kernel='—' if bk=='cpu' else f"{b['resident_kernel_seconds']:.6f}"
   text+=f"| {s['B']} | {bk} | {b['total_seconds']:.6f} | {kernel} | {b['eigenvalue_max_abs_error']:.3g} | {b['independent_full_H_max_residual']:.3g} |\n"
 text+='\n| 系统数 | GB10 seq→batch端到端倍数 | AMD seq→batch端到端倍数 | 优化NumPy全eigh参考秒 |\n|---:|---:|---:|---:|\n'
 for s in d['systems']:text+=f"| {s['B']} | {s['batch_speedups']['cuda']['end_to_end']:.3f} | {s['batch_speedups']['hip']['end_to_end']:.3f} | {s['metadata']['CPU_numpy_full_eigvalsh_seconds']:.6f} |\n"
 text+='''
新seq与batch**两者都驻留GPU**；seq故意一次只发一个k点block，batch一次grid并行全部k点。因此高倍数主要反映此对照的并行占用率，不是官方Daviator22倍复现，不能把全部收益归因PCIe消除或声称通用硬件排名。旧混合版也采用不同子空间/重启策略，不能只把两个倍数并排就说改了一行得到几十倍。

优化NumPy参考B64约0.229s，仍快于两GPU完整启动batch的0.457/0.378s；B256约0.906s，GPUbatch为0.791/0.698s。教学CPU C++保留作另一参考，不用慢CPU隐藏这个事实。

全部B64×16和B256×16本征值对独立eigvalsh误差<=7.22e-15，小于1e-13；独立完整H残差<=9.90e-11、正交性也通过，CUDA/HIP特征值逐值相同。全部工作在原合成周期余弦势，不冒称真实硅自洽DFT。

`results/module10_results.json`保留旧hybrid字段并嵌入`resident_upgrade_module15`，新完整账本为`results/module15_results.json`。**图9现更新为驻留版**；前面第三阶段的hybrid表是历史对照，旧图另留`fig9_daviator_hybrid_original.svg`。

![全GPU驻留Davidson](visuals/fig9_daviator_batched_davidson_vasp_band.svg)

## 模块十六：381逐字数字与事实终审（正文未改）

'''
 text+=f"正式审稿意见在`{a['review_path']}`；本文仅记录审稿完成，不代表原稿已经修正或批准发布。\n\n"
 text+=f"- P0 {a['findings_by_severity']['P0']}项：不能把全部CUDA-X一刀切成无Tensor Core；不能把源码迁移、机制原型和完整模型浮点容差都叫原封不动/逐字节一致。\n"
 text+=f"- P1 {a['findings_by_severity']['P1']}项：包括硝苯地平25重原子/25×25、Uni-Mol实际31词元、Gaussian归一化分母、Dummy均值-3.06、BBBP1955去重→1952构象成功、先AddHs、计时/盐示例来源、训练集遮挡解释、BBB机制/嗜睡措辞与封面裁切。\n"
 text+=f"- P2 {a['findings_by_severity']['P2']}条局部建议，保留作者口气，不要求重写。\n"
 text+='- 七种药物CID与21描述符、全部ESOL/BBBP/GB1成绩表、原CPU计时均有实录；正确数字明确标通过，不因复跑时间不同而误改旧快照。\n- 7个唯一外链中4个GitHub内容通过；3个微信HTTP200但跳验证码，远端正文未能确认，不当死链也不冒称读到正文。\n'
 text+=f"\n381正文初末SHA256均为`{a['article_sha256_after']}`；五SVG和封面初末哈希也一致，未写入。审稿意见SHA256 `{a['review_sha256']}`。读写范围严格限定审稿意见，未将建议直接套进正文。\n\n"
 text+='''
## 第四阶段入口、验收和同步

```bash
.venv/bin/python run_pipeline.py --phase 4
.venv/bin/python run_pipeline.py --phase 4 --reuse-results
.venv/bin/python verify_phase4.py
```

`audit381_record`是核对**已有人工/Agent终审**的正文身份与报告哈希，不能凭Python命令自动重做联网和逐字审稿；正文变化会拒绝复用并要求重新终审。`--reuse-results`只重验已执行产物，所有真实GPU日志另留。

`verify_phase4.py`从10k完整历史/坐标/拓扑重算单调性、键误差、同一碰撞定义，重核64/256全部本征对/残差/正交性和零主机往返元数据，以及381/审稿意见/配图哈希。旧弱排斥失败记录保留并参与验收，不能只挑成功均值。

源码、轻量结果、图9/13和审稿记录同步实验双仓；大坐标、历史与H数据留本地，种子/生成脚本/哈希随Git。公众号仓仅新增审稿意见，原正文与配图不变。
'''
 return text
if __name__=='__main__':print(report_text())
