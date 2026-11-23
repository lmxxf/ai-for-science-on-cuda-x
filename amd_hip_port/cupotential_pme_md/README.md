# Original PME + flexible MD prototype (MIT)

`pme_md.cpp` is original MIT C++ compiled with CUDA/cuFFT or HIP/hipFFT; CPU uses matching independent NumPy/SciPy equations in repository-root `exp7_pme_and_fused_md_hip.py`. This is not libcupo, not OpenMM API-compatible, not production SETTLE/PME code.

## Implemented physical model

27 synthetic neutral flexible trimers / 81 atoms in a 24Å periodic cubic box. Charges (+0.08,-0.16,+0.08)e, masses(12,16,12)amu; bonds k=200kJ/(mol Å²), r0=2Å; angle k=30kJ/(mol rad²), theta0=1.9rad. All atom pairs including intramolecular pairs have LJ sigma1.2Å/epsilon0.05kJ/mol and Ewald electrostatics; no bonded exclusions or constraints are implied. These are deliberately defined synthetic molecules, not water or experimental observations.

PME: fourth-order cardinal B-spline spreads charge to 64³ mesh, full complex-to-complex double-precision 3D FFT, Green factor 4pi exp(-k²/4alpha²)/k², continuous B-spline sinc^8 deconvolution in each dimension, k=0 omitted (neutral conducting/tin-foil boundary). Inverse unnormalized FFT gives mesh potential after ke/V scaling. Reciprocal energy is 1/2 sum rho*phi. **Forces use derivatives of B-spline interpolation**, the derivative of that same mesh energy. An ik-field interpolation, while common, would not be exactly the derivative of this discrete energy; it is intentionally not mixed in here.

Real space is all-pairs MIC erfc(alpha*r)/r plus LJ up to10Å, alpha0.4Å^-1. All-pairs is deliberate for81 atoms; no claim of a new large-system cell-list speedup. Self energy -ke alpha/sqrt(pi) sum q² is included. No reciprocal surface correction for neutral tin-foil conditions. LJ unshifted at cutoff; erfc tail at10Å is tiny but finite. PME force error versus direct Ewald is measured, not assumed zero.

Units: positionsÅ, timeps, massamu, energykJ/mol, chargee; ke1389.354576kJ Å/(mol e²), kB0.00831446261815324kJ/(mol K). Kinetic=.005 sum m v²; acceleration=100F/m.

## Dynamics and device residency

NVE Velocity-Verlet1000steps dt0.5fs to0.5ps, and2000steps dt0.25fs to the same physical end time for second-order convergence. NVT BAOAB1000steps dt1fs, gamma10ps^-1, target300K; reproducible per-coordinate/step 64-bit hash + Box-Muller noise.

GPU coordinates/velocities/force/FFT mesh and diagnostics stay on device during each entire host-submitted integration loop. Fused kick/drift and kick kernels execute on GPU; this is not a single persistent kernel or a claim of CUDA Graph optimization. Every step recomputes full real+reciprocal+self+bond+angle force. No per-step host readback. Diagnostic kernel stores energy, kinetic, total, temperature, first bond length and x-coordinate of atom0 each step. Full final positions and velocities are also saved; the prototype does **not** dump every atom's full XYZ history.

## Validation / limitations

Independent direct Ewald in Python explicitly sums neighboring real image boxes, reciprocal k=-16..16 (and checks18), and self energy; LJ/bond/angle terms use the stated model. This has no PME mesh. Mesh energy gradients are independently central-finite-differenced, NVE step-halving is measured, and net force residual is reported. Finite mesh breaks continuous translational invariance: nonzero mesh net-force residual is visible and decreases with refinement. It is not hidden behind cross-vendor numerical agreement.

Full complex FFT was used for simple consistent vendor interfaces rather than an optimized R2C/C2R packing. CPU NumPy/SciPy workers=1 is a clear baseline, not the same compiled implementation or an optimized CPU PME engine. Single trajectory wall times are actual runs, not medians of repeated benchmarks.

## Reproduce

From repository root:

```bash
.venv/bin/python guard_process.py --audit logs/module9_all_memory.json -- .venv/bin/python exp7_pme_and_fused_md_hip.py
```

`--cpu-only` prepares data/independent physics andCPU trajectories; `--gpu-only` recompilesCUDA andruns bothGPUs on prepared inputs; `--summarize-only` re-reads existing binaries/JSON and redraws. Local GPU subprocesses and compilation are guarded. Remote directory `C:\AI4S-HIP-Lab\cupotential_pme_md` is isolated and only reads the existing ROCm/VS install. `build_windows.cmd` links hipFFT; `run_windows.cmd` executes the model. CUDA needs `-lcufft`. All double precision, contractiondisabled, nofastmath.

Raw outputs: `results/module9_{cpu,cuda,hip}`; initial forces, 1001/2001-row historybinary+CSV, finalpositions/velocities; summary `results/module9_results.json`; independent validation `results/module9_validation.json`.
