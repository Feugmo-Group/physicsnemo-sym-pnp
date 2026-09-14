# physicsnemo-sym-pnp

PhysicsNeMo Sym implementations of eleven physics-informed neural network (PINN) strategies
for the one-dimensional Poisson–Nernst–Planck (PNP) system of a lithium symmetric cell.

This is the code for:

> D. Pankaczy and C. G. Tetsassi Feugmo, *A Systematic Benchmark of Physics-Informed Neural
> Network Strategies for Stiff Ion Transport in Lithium Symmetric Cells*, Journal of The
> Electrochemical Society (submitted).

## The problem

Dimensionless PNP for a binary LiPF6 electrolyte between two lithium electrodes:

```
-eps^2 d2phi/dx2 = z_p c_p + z_n c_n
   dc_p/dt = d2c_p/dx2 + z_p d/dx (c_p dphi/dx)
   dc_n/dt = xi [ d2c_n/dx2 + z_n d/dx (c_n dphi/dx) ]
```

on `(x, t) in [0, 1] x [0, 2.56]`, with a flux boundary condition at both electrodes.

The dimensionless groups are **derived at run time** from the physical constants in
`registry/parameters.py`; nothing is hard-coded:

```python
from registry import Parameters
p = Parameters()
p.eps      # 3.751812551810829e-07   Debye length / cell length
p.xi       # 10.0                    D_n / D_p
p.delta    # 0.38866011260654626     dimensionless applied current
p.t_c      # 1406.25 s               characteristic time
```

`eps^2 = 1.41e-13` is what makes the system stiff, and what the benchmark is about.

## Layout

```
conf/            one Hydra config per benchmarked configuration
registry/
  parameters.py  physical constants and derived dimensionless groups
  models.py      custom FBPINN, KAN and SPINN architectures
  loss.py        BRDR (balanced residual decay rate) loss aggregator
  geometry.py    grid-based sampling for the separable architecture
pnp.py           main driver (eight configurations)
pnp_decoupled.py one network per field
pnp_symmetric.py symmetric / antisymmetric variable transformation
fvm/
  pnp_fvm.py     finite-volume reference solver
  pnp.zip        its output, pnp.csv, used as the validation reference
```

## Running

Everything was run in the NVIDIA container, which pins the software stack:

```bash
docker run --rm --init --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user="$(id -u):$(id -g)" --volume="$PWD:/app" --workdir=/app \
  nvcr.io/nvidia/physicsnemo/physicsnemo:25.11 \
  bash -c "python pnp.py --config-name=config_ntk"
```

The configuration is always selected explicitly with `--config-name`:

| Configuration | Command |
|---|---|
| Vanilla PINN | `python pnp.py --config-name=config` |
| NTK weighting | `python pnp.py --config-name=config_ntk` |
| BRDR weighting | `python pnp.py --config-name=config_brdr` |
| AdaHessian | `python pnp.py --config-name=config_adahessian` |
| Fourier features | `python pnp.py --config-name=config_fourier` |
| PIKAN | `python pnp.py --config-name=config_kan` |
| FBPINN (multilevel) | `python pnp.py --config-name=config_fb` |
| SPINN | `python pnp.py --config-name=config_separable` |
| Enriched PINN | `python pnp.py --config-name=config_enriched` |
| Decoupled PINN | `python pnp_decoupled.py --config-name=config` |
| Sym./antisym. transform | `python pnp_symmetric.py --config-name=config` |

Unzip `fvm/pnp.zip` first; the validator reads `fvm/pnp.csv`.

Each configuration in the paper was run ten times as a SLURM array job, with
`hydra.run.dir=./outputs/<name>_${SLURM_ARRAY_TASK_ID}`.

## Training protocol

Common to every configuration: 100,000 optimizer steps, gradient accumulation over 4
forward/backward passes, gradient-norm clipping at 0.5, Adam at 3e-4 with exponential decay
(0.92 every 4000 steps), 16,000 interior and 1,600 boundary collocation points resampled at
every step, and a validation RMSE against the FVM reference every 1,000 steps.

Exceptions: AdaHessian halves the collocation budget (8,000 / 800) to fit the
Hessian-vector product and uses a decay rate of 0.95; NTK recomputes its weights every
10 steps (`training.ntk.run_freq`); SPINN samples on a tensor-product grid, as its separable
ansatz requires.

The ten runs per configuration differ only in PyTorch's default weight initialisation; they
are not seeded. Per-run results are reported rather than a single reproducible run.

## Reproducing the reference solution

```bash
python fvm/pnp_fvm.py
```

Method of lines on 500 uniform nodes (h = 1/499), Poisson solved algebraically at each step,
Radau time integration with rtol 1e-6 and atol 1e-8, written to `pnp.csv` as a
3600 x 500 space-time grid for each of `cp`, `cn`, `phi`.

## Citing

The paper cites this repository at commit `a8e5034`. Please refer to that commit rather than
the repository head, so that the version is unambiguous.

## License

Apache License 2.0. Portions derive from NVIDIA PhysicsNeMo Sym and retain their original
copyright headers.
