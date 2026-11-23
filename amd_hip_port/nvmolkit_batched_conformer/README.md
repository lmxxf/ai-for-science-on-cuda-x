# Original batched united-heavy-atom conformer optimizer (MIT)

`conformer.cpp` is original same-source single-thread C++ / CUDA / HIP code. It is **not nvMolKit or MMFF94**, does not reproduce their atom typing or parameterization, and is not a drop-in API replacement. It illustrates a block-per-conformer GPU optimization mechanism on real ESOL molecular topology.

## Actual inputs and defined model

The first1000 rows of local cleaned ESOL, ten **fresh ETKDGv3** conformers each. Initial seed20261124+row; no MMFF/UFF optimization is performed. Explicit deterministic iid sigma0.06Å coordinate noise is added after embedding to test relaxation from bond strain. Topology is heavy atoms; explicit hydrogens are used for ETKDG/charge derivation, then their Gasteiger charges are folded into bonded heavy atoms. The code never reads an old optimized coordinate cache.

Equilibrium bond lengths and angle targets are read from RDKit MMFF parameter APIs where available; no MMFF optimization is called. Unsupported equilibrium geometries fall back to documented covalent-radii/hybridization rules. These equilibrium targets alone **do not make this MMFF94**.

All energies in kJ/mol, distancesÅ; parameters define a deliberately simplified united-heavy-atom model:

- Bond: `E=.5*400*(r-r0)^2`.
- Cos-angle: `E=.5*40*(cos(theta)-cos(theta0))^2`.
- Torsion: normals n1=(r0-r1)×(r2-r1), n2=(r2-r1)×(r3-r2), c=n1·n2/(|n1||n2|). Period2 `E=.2*(1-cos(2phi))=.4*(1-c²)`; period3 `E=.2*(1+cos(3phi))=.2*(1+4c³-3c)`. This phase/convention is explicitly defined, not a claim of MMFF-derived torsional profiles or chemically calibrated minima. Norm squared gets1e-16 regularization near degenerate planes.
- Nonbonded for all graph-distance>=3 pairs including1–4: LJ12–6 epsilon0.5, sigma_ij=.85*(Rvdw_i+Rvdw_j) plus `1389.354576*qi*qj/r`. Both use softcore `r=sqrt(|ri-rj|²+.01)` for finite gradients. No Coulomb cutoff, screening, PME or special1–4 scaling. No explicit hydrogen optimization.

Parameter selection and failure are retained: initial epsilon0.05 let Coulomb-driven contacts grow from47 to1138 despite energy decrease. Epsilon was explicitly increased to0.5 on identical initial coordinates; this is a prototype correction, not parameter validation on independent chemistry. Failed raw outputs and summary live under `results/module14_failed_weak_lj*`. No collision threshold was changed to hide the failure. The criterion is graph distance>=3 and physical separation<0.8*sigma; it is a declared diagnostic, not a universal chemical clash definition.

## Algorithm / memory / determinism

One64-thread block per conformer, at most64heavy atoms (actual maximum55). Coordinates, candidate coordinates, gradients and per-atom energies stay in shared memory for200steps. Parameters/topology are **read from global memory**, and per-step energies are written to a global diagnostic buffer. Thus this is **not zero global access**. Only state iterates remain resident, with no per-step CPU intervention.

Each atom gathers its incident force-field terms in a fixed order; each potential contributes E/arity to its incident atoms. Atom energies and maximum gradient are then reduced in fixed atom order by thread0. No floating-point atomics. Initial step.003, gradient clipping limits atom gradient norm to10, up to16 backtracking trials accept non-increasing energy; accepted step×1.15 capped.02, rejected step×.5. This is200descent attempts, not a guarantee of a stationary point or global optimum. Final gradient norms are reported.

CPU performs the exact same gather/order/state arithmetic serially. CUDA/HIP use float64 with contraction disabled, nofastmath. Cross-platform comparison covers all400,350 active coordinates plus final10000energies; padding is zero-initialized too.

## Verification and evidence

The driver exports actual coordinates/topology/explicit parameter binaries. An **independent NumPy energy expression** is finite-differenced with five-point stencil h2e-4Å on18directions across three real molecules, against the C++ analytic gradient. Full10000×201energy histories verify monotonicity; bond-deviation and declared contact counts are computed on allconformers. Energy descent alone is not used as a chemical-quality guarantee.

`results/module14_results.json` includes final timings, all checks, exact difference counts, preparationtime and the genuine olderRDKit timings only as a **different-task reference**. The prior66.608s/9.987s included ETKDG+actualMMFF94; it is invalid to divide them by this prototype runtime and claim equivalent MMFF/nvMolKit acceleration. Fair throughput comparison is against the same-source C++ model; fresh ETKDG+parameter preparation is separate.

## Reproduce

From repository root:

```bash
.venv/bin/python guard_process.py --audit logs/module14_all_memory.json -- .venv/bin/python exp12_open_nvmolkit_batched_conformer_hip.py
```

`--stage prepare` rebuilds freshETKDG/topology; `--stage run` compiles, validates gradients and runsCPU/CUDA/HIP; `--stage gpu` reruns bothGPUs on an existing CPU/input baseline; `--stage summarize` only reads existing evidence and redraws. Local heavy tasks are guarded. Independent remote directory `C:\AI4S-HIP-Lab\nvmolkit_batched_conformer` only reads existingROCm/VS; no Strata modifications.

Binary format is documented by Python dtype and C++ staticassert: Mol fourint32 `(n,start,count,atomstart)`, Term sixint32+fourfloat64 (56bytes), incident-term CSR arrays, paddedfloat64 initialcoords `(10000,stride,3)`. Outputs coordinates, maxgradient andenergyhistory `(10000,201)`, plus actual single-complete-run walltime/H2D/D2H JSON.
