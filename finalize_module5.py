#!/usr/bin/env python3
from pathlib import Path
import json,datetime,hashlib
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
def main():
 amd=load('results/module5_amd9070/timings.json');nv=load('results/module5_gb10_cpp/timings.json');original=load('results/module3_results.json');comparison=load('results/module5_comparison.json')
 assert comparison['all_byte_identical'] and comparison['total_mismatches']==0
 r={'timestamp':datetime.datetime.now().astimezone().isoformat(),'seed':20261123,'cells':50000,'points':2000000,'genes':500,'dtype':'float64','workload':'identical exported synthetic spatial-omics-scale inputs; not experimental tissue','amd':amd,'gb10_same_harness':nv,'gb10_original_python_pipeline_ms':original['gpu_pipeline_median_s']*1000,'cpu_original_pipeline_ms':original['cpu_pipeline_median_s']*1000,'comparison':comparison,'input_manifest':load('results/module5_data/header.json'),'cpu_to_amd_device_resident_speedup':original['cpu_pipeline_median_s']*1000/amd['pipeline_median_wall_ms'],'amd_over_gb10_same_harness_time_ratio':amd['pipeline_median_wall_ms']/nv['pipeline_median_wall_ms'],'amd_h2d_plus_resident_estimate_ms':amd['h2d_ms']+amd['pipeline_median_wall_ms'],'amd_h2d_pipeline_d2h_estimate_ms':amd['h2d_ms']+amd['pipeline_median_wall_ms']+amd['d2h_ms'],'notes':['Exact same binary inputs, all integer output arrays byte-identical; intermediate floating-point states need not be bit-identical','No game executable detected before GPU benchmark; Steam/trainer/emulator background processes not terminated','Native harness retains original allocation and host scalar synchronization; medians of separately timed stages do not add to full pipeline median','CUDA nvcc --fmad=false; HIP clang -ffp-contract=off; no fast math','GPU timings use 3 warmups and 7 repeats; host wall and event arrays both retained','Original Python GB10 9.64ms and native C++ timings are separate entrypoint scopes; same-harness comparison is primary','Molecular-potential bonus not implemented; this is cuSpa HIP portability validation, not a cuPotential replacement']}
 (R/'results/module5_amd9070_results.json').write_text(json.dumps(r,indent=2)+'\n')
 print('AMD result finalized:',amd['pipeline_median_wall_ms'],'ms, mismatch',comparison['total_mismatches'])
if __name__=='__main__':main()
