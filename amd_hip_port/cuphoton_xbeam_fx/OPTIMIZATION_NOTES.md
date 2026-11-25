# FX full-workload CUDA/HIP optimization, 2026-11-25

Original source, driver, old JSON and all three original120MiB visibility arrays are preserved under `results/module11_optimized_previous/` with SHA256 manifest. The first1warm+3repeat trial is also retained: `results/module11_optimized_trial_results.json`, `results/module11_optimized_{cuda,hip}/`, and trialsource/driver in the archive. Final evidence is `results/module11_optimized_results.json`, using `results/module11_optimized_stable_{cuda,hip}/`.

No change to `results/module11_results.json`, DevHistory or images was made by this task. Parent merges historical/latest measurements separately.

## Same computation / source / configuration

All67,108,864 inputcomplex128 samples (16antennas ×4096frames ×1024FFT) are processed, retaining64timebins,64frames/bin,120baselines,1024channels,7,864,320 complex outputvisibilities. InputvoltageSHA256 is checked remotely before launch. No resampling, lowered precision, truncated channels or selected baselines.

Onecommon `amd_hip_port/cuphoton_xbeam_fx/fx.cpp` is compiled for bothCUDA sm_121 andHIP gfx1201. Flags: CUDA `-O3 --fmad=false`; HIP `-O3 -ffp-contract=off`. No fastmath/TF32/tensorcores. All X summations retain chronological t=0..63 order, original complex multiply/add expression and final division64.

Threefixed variants are benchmarked on both cards:

- **v0 control:** original1024thread radix2 FFT with kernelcos/sin; original onebaseline-channel perthread X, adapted only to a fullresident workload harness.
- **v1:** hostprecomputed512 complex twiddles (one table perprogram), FFT1024threads uses table reads; X onechannel/anchorantenna thread accumulates remaining antenna pairs in registers, reusing anchorload without changing t-order.
- **v2:** tableFFT512threads (twoinput/output entries perthread, same butterfly order), X2channel×64frame×16antenna sharedpanel (32KiB);240threads of256 map120pairs×2channels, reusing panel data across baselines. Exactly the same tile and thread configuration on both cards.

There was one moderate tile attempt, not an unbounded search. v2 is selected **as one common variant for both cards** by geometricmean paired performance; percard-best variants are not mixed.

## Correctness on every output

New CPU baseline also performs a complete resident-data warmup followed by3complete measured passes, using the same NumPy FFT and X formulas as the original reference. CPUresident FX wallmedian1.056031693s (1.055783196–1.056401181s), FFTmedian0.207233561s, Xmedian0.848999587s. CPUdiskread is separate.

All six GPU variant/vendor visibility arrays are compared with the full CPU array; relativeL2 max4.42655e-16 <1e-14. v0 matches the historical raw output from its corresponding GPU **bitwise**. TableFFT variants differ at ~2.6–2.8e-16 from historicalGPU due to hosttwiddle versus devicetrig rounding, not a changed mathematical workload.

The existing commonCPU gridding/IFFT/HögbomCLEAN path is applied to all complete outputvisibility arrays. RelativeL2 versus CPUreferenceimage max7.96683e-16 <1e-14. Selectedv2 crossGPUvisibilityL2=2.95811e-16, commonCPUCLEANimageL2=6.72932e-16. CLEAN is not secretly performed onGPU and its time is not included in FX.

## Final warm5 + repeat7 measurements

First trial revealed AMD timing drift. Without changing any kernel or tile, finalsource was recompiled on **both** cards with5complete workloadwarmups and7measured repetitions pervariant. Allwarmupwall values and raw7stagewall/deviceevent/combinedwall/deviceevent arrays are inJSON. The older1+3records remain intact. CPU was already stable and was not rerun for the finalGPUonly stability check.

### Common selected v2, fullresident FX (ms)

| Backend | F wallmedian | X wallmedian | FX wallmedian | FX min–max |
|---|---:|---:|---:|---:|
| NVIDIA GB10 |17.435406|22.205913|39.637526|39.633638–39.777769|
| AMD RX9070XT |8.963900|11.056700|19.809200|19.730700–21.024100|

CPU1.056032s is a NumPy/C implementation baseline; GPU kernel execution is nativeCPP. Those are documented rather than claimed equivalent software stacks.

GB10v2 CV0.15%; AMDv2 CV2.15%, max-min/median6.53%. **This is not a claim of locked clocks or perfectly flat AMD timing.** FinalAMDv0control also varies113.0189–127.4689ms (median125.2934ms, CV4.27%). Raw ranges remain visible; no fastest-only run or later besttail was selected. No more tuning or repeats were performed after this requested stability check.

### Same-harness control and intermediate results, FX median(ms)

| Backend |v0 originalkernel|v1 table/register|v2 table/sharedpanel|
|---|---:|---:|---:|
| NVIDIA GB10 |133.761674|59.557605|39.637526|
| AMD RX9070XT |125.293400|19.820500|19.809200|

Sameharness v0→v2 speedup3.3746× onGB10 and6.3250× onAMD, with AMD range caveat above. Historical64-bin/singlepass/disk-streaming JSON is not used as the controlled denominator.

**Negative result retained:** Xalone did not materially beat the originalX. GB10v0X20.789ms versusv1register37.970ms andv2panel22.206ms. AMDv0X10.910ms versusv1register10.508ms andv2panel11.057ms (variation is non-negligible). The strong totalFX gain is dominated by removingFFTtrigonometry and improvingFFTblock utilization, not evidence that all reuse strategies are faster. Global-memory reuse can trade against registerpressure/sharedbank/coalescing costs.

## Timing scope and resource behavior

GPU input1GiB and spectrum1GiB plus120MiB output buffers remain resident during allmeasured kernels. This changes batching from historical64separate diskstreamed calls, so v0 was rebenchmarked under exactly the same newharness. Original v0numeric outputs are identical, confirming unchanged sum semantics.

Hostwall and GPUevent timings are distinct. FFT and X stage medians are independently sampled; their sum need not equal separately timed FXmedian. Allvariants-programwall includes manywarmups/repeats and I/O; it is **not** a onepass application end-to-end latency. H2D/D2H andinput/outputdiskI/O are separate. `host_transfer_plus_pipeline_estimate_seconds` is explicitly a sum estimate, not a measuredapplicationloop or CLEAN-inclusive result. Code reports real values without hiding transfers; onPCIeAMD they can reverse the devicecompute ordering relative to unifiedmemoryGB10.

Win32_Process full lists were read before remote use and checked again immediately before launch; Steam/trainer/MuMu backgrounds were present but no gamebinary was identified. No userprocess was killed. ExistingROCm/VS/Strata files were only read; workremains in isolated `C:\AI4S-HIP-Lab\cuphoton_xbeam_fx`.

All Spark heavy processes were run through250ms/20GiB memoryguard, with a separate auditJSON percompile/CPU/GPU/fullrun. A PowerShelllocalizedstderr UTF8decode failure happened after successful first localGPU/CPU runs; decoding was corrected to `errors='replace'` and only the remote stage resumed, avoiding duplicate local measurements. This was a hostorchestration failure, not a GPUcorrectness failure.

## Reproduce / artifacts / source sync

Prepared1GiBinput and historicalreferencefiles are prerequisites for this specificoptimization audit. To rerunCPU+GPUfinalprotocol: repositoryroot `.venv/bin/python exp9_cuphoton_blackhole_fx_hip.py`; `--gpu-only` reuses alreadymeasuredCPUreference; `--summarize-only` only reads existing finalartifacts and writes the independentoptimizedsummary. It deliberately does not overwrite originalcanonicalJSON or figures.

NativeCLI: `fx DATA_DIR OUTPUT_DIR 7 5` =7repeats,5warmups. Remote uses `build_optimized_windows.cmd` and `run_optimized_windows.cmd`. OriginallegacyCLI/files are retained in the archive, not silently reused as the CUDA side of a HIPonly optimization.

Raw outputfiles remain locally underresults; they are not to be committed asGit blobs. The newsource, driver, compile/run scripts, notes, and smallJSON evidence are synced to both experimentrepositories; SHA256manifest verifies identicalcommon source anddriver copies. Parent owns finalGit staging/commit decisions.
