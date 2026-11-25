# Open nonbonded CPU / CUDA / HIP


2026-11-25布局对照已完成：默认保留v0，AoS/SoA候选的AMD局部收益与GB10回退见[最新优化记录](OPTIMIZATION_NOTES.md)。下面原计时表为初版基线；完整复测入口：`python exp5_open_cupotential_hip.py --optimized`（在仓库根目录）。

Original MIT implementation for this experiment. **Not NVIDIA cuPotential, not libcupo source, not API-compatible, not a complete molecular dynamics engine.** `open_nonbonded.cpp` is the same source compiled for CPU, CUDA or HIP; vendor primitives are CUB/hipCUB, which remain separately licensed dependencies.

## Physics and scope

Units: coordinates and sigma in Å, charge in elementary e, epsilon and energy in kJ/mol, force in kJ/(mol Å). Coulomb conversion is 1389.354576 kJ Å/(mol e²). Cubic PBC uses `d -= L*floor(d/L+0.5)` minimum image. Cutoff is 10Å. Cell width is at least cutoff, each axis has at least 3 cells, so 27 periodic neighbors contain every candidate exactly once. Grid binning, stable radix sort, scans and all pair evaluations execute on GPU. CPU is single-thread compiled C++ with stable counting buckets and the same pair expression.

Lorentz-Berthelot: sigma_ij=(sigma_i+sigma_j)/2, epsilon_ij=sqrt(epsilon_i epsilon_j).

```
E_LJ = 4 epsilon_ij ((sigma_ij/r)^12 - (sigma_ij/r)^6)
k_rf = (epsilon_RF-1)/(2 epsilon_RF+1) / rc^3
c_rf = 3 epsilon_RF/(2 epsilon_RF+1) / rc
E_RF = k_e qi qj (1/r + k_rf r^2 - c_rf)
F_i += [24 epsilon_ij (2 sr12 - sr6)/r^2
        + k_e qi qj (1/r^3 - 2 k_rf)] (ri-rj)
```

Outer medium dielectric epsilon_RF=78.3; this is an assumed reaction-field boundary condition, not explicit solvent or a proof the synthetic atoms model water. Total system charge is neutral. There are no bonded exclusions, 1–4 scaling, constraints, integrator, reciprocal-space PME, thermostat or barostat. LJ is **unshifted hard-truncated**, hence energy and force discontinuities at cutoff; RF energy is shifted to zero at cutoff but the finite-dielectric force is not strictly zero. Finite differences avoid cutoff crossings. Timing is a nonbonded evaluation (index+force+energy), not an MD step or ns/day.

Each atom is assigned one thread. Both i→j and j→i are evaluated, energies are divided by two, forces are not. Per-atom outputs avoid atomic floating-point reduction; the total is accumulated on host after timing. Each GPU uses float64, contraction disabled, no fast-math. Neighbor atom order within cells is stable and shared across CPU/CUDA/HIP.

## Data, validation and execution

Inputs are fixed-seed jittered lattice systems of 32,000 and 128,000 synthetic atoms with bounded charges, sigma and epsilon; minimum separation is bounded below by 3.2Å, preventing divergent near-zero pair interactions. This is **not experimental biological data or a validated solvated molecular topology**.

`atoms.bin`: little-endian float64 N×6 `(x,y,z,q,sigma,epsilon)`; `params.txt`: `N box_length cutoff epsilon_RF`. Outputs `energy.bin` float64 N (half pair energy summed for each atom), `forces.bin` float64 N×3, plus `timings.json`.

The repository-root Python driver `exp5_open_cupotential_hip.py` generates inputs, compares all energy/force values, compiles each backend, and renders figure6. Invoke from repository root:

```bash
.venv/bin/python exp5_open_cupotential_hip.py --prepare-only
# Once module six GPU work has finished:
.venv/bin/python exp5_open_cupotential_hip.py --gpu-only
```

All local CPU/GPU workloads are wrapped by the repository memory guard. Remote Windows workspace is `C:\AI4S-HIP-Lab\cupotential_open`; build scripts only read existing ROCm and Visual Studio installations. Never modify Strata sources or its environment. AMD GPU execution in this experiment has explicit exclusive-use permission.

Independent correctness tests in Python: 216-atom O(N²) NumPy brute-force versus C++ cell list, six finite-difference force components from independently perturbed **brute-force energies**, near-zero net force, and periodic translation invariance. Full 32k/128k force arrays and total energies then compare CPU↔CUDA↔HIP with target relative L2<1e-12. Shared CPU/GPU code alone is not treated as proof of physical correctness.
