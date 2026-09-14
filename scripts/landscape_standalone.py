"""Loss-landscape scan and loss variation ratio S for the 1D PNP benchmark, PyTorch only.

Reimplements the PhysicsNeMo Sym fully-connected architecture (weight-normalised linear
layers, SiLU activations) and the dimensionless PNP residuals directly, so the scan can be
reproduced from a checkpoint with nothing but PyTorch installed. Results are identical to
running the same scan inside the PhysicsNeMo container; the container version lives in
landscape_sharpness.py and additionally handles the custom FBPINN, KAN and SPINN
architectures.

    python landscape_standalone.py --run-dir outputs/pnp_ntk_1 --range-scale 0.01

Writes <out>.npz with the raw surfaces and prints S per loss component and for the total.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

# Physical parameters, identical to registry/parameters.py
C0, DP, DN, I_APP, L = 500.0, 4.0e-10, 4.0e-9, 10.0, 7.5e-4
T, EPS0, EPSS, R, F = 298.15, 8.85e-12, 16.8, 8.314, 96485.332
ZP, TF = 1, 3600.0

EPS = float(np.sqrt(R * T * EPSS * EPS0 / (ZP**2 * F**2 * C0 * L**2)))
XI = DN / DP
DELTA = I_APP * L / (ZP * F * C0 * DP)
TC = L**2 / DP
YF = TF / TC


class WeightNormLinear(nn.Module):
    """PhysicsNeMo Sym's weight-normalised linear layer: W = g * V / ||V||_row."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        self.weight_g = nn.Parameter(torch.empty(n_out, 1))
        self.bias = nn.Parameter(torch.empty(n_out))

    def forward(self, x):
        w = self.weight_g * self.weight / self.weight.norm(dim=1, p=2, keepdim=True)
        return torch.nn.functional.linear(x, w, self.bias)


class FullyConnected(nn.Module):
    def __init__(self, n_in=2, n_out=3, width=512, depth=6, act="silu"):
        super().__init__()
        self.layers = nn.ModuleList()
        d = n_in
        for _ in range(depth):
            self.layers.append(WeightNormLinear(d, width))
            d = width
        self.final = nn.Linear(width, n_out)
        self.act = {
            "silu": torch.nn.functional.silu,
            "elu": torch.nn.functional.elu,
            "tanh": torch.tanh,
        }[act]

    def forward(self, x):
        for lyr in self.layers:
            x = self.act(lyr(x))
        return self.final(x)

    def load_physicsnemo(self, sd):
        m = {}
        for i in range(len(self.layers)):
            for s in ("weight", "weight_g", "bias"):
                m[f"layers.{i}.{s}"] = sd[f"_impl.layers.{i}.linear.{s}"]
        m["final.weight"] = sd["_impl.final_layer.linear.weight"]
        m["final.bias"] = sd["_impl.final_layer.linear.bias"]
        self.load_state_dict(m)


def d1(u, v):
    return torch.autograd.grad(u, v, torch.ones_like(u), create_graph=True)[0]


def residuals(net, x, y, symmetric=False):
    """Dimensionless PNP interior residuals, matching pnp.py / pnp_symmetric.py."""
    out = net(torch.cat([x, y], dim=1))
    a, b, phi = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    if symmetric:
        c, rho = a, b
        cp, cn = c + rho, c - rho
    else:
        cp, cn = a, b

    cp_x, cn_x, phi_x = d1(cp, x), d1(cn, x), d1(phi, x)
    cp_xx, cn_xx, phi_xx = d1(cp_x, x), d1(cn_x, x), d1(phi_x, x)
    cp_y, cn_y = d1(cp, y), d1(cn, y)

    poisson = EPS**2 * phi_xx + (cp - cn)
    cont_p = cp_y - (cp_xx + cp * phi_xx + cp_x * phi_x)
    cont_n = cn_y - XI * (cn_xx - cn * phi_xx - cn_x * phi_x)
    return poisson, cont_p, cont_n


def boundary(net, x, y, symmetric=False):
    out = net(torch.cat([x, y], dim=1))
    a, b, phi = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    cp, cn = (a + b, a - b) if symmetric else (a, b)
    cp_x, cn_x, phi_x = d1(cp, x), d1(cn, x), d1(phi, x)
    return {
        "flux_cp": -cp_x - cp * phi_x - DELTA,
        "flux_cn": -cn_x + cn * phi_x,
        "phi_x": phi_x,
        "phi": phi,
    }


def halton(n, dim, skip=20):
    primes = [2, 3, 5, 7][:dim]
    out = np.empty((n, dim))
    for d, base in enumerate(primes):
        for i in range(n):
            f, r, k = 1.0, 0.0, i + 1 + skip
            while k > 0:
                f /= base
                r += f * (k % base)
                k //= base
            out[i, d] = r
    return out


def make_closure(net, n_int, n_bnd, device, symmetric):
    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device, requires_grad=True)

    p = halton(n_int, 2)
    xi_, yi_ = T(p[:, 0:1]), T(p[:, 1:2] * YF)
    q = halton(n_bnd, 1, skip=37)
    x0, y0 = T(q), T(np.zeros((n_bnd, 1)))
    xl, yl = T(np.zeros((n_bnd, 1))), T(q * YF)
    xr, yr = T(np.ones((n_bnd, 1))), T(q * YF)

    def closure():
        po, cp_, cn_ = residuals(net, xi_, yi_, symmetric)
        pde = (po**2).mean() + (cp_**2).mean() + (cn_**2).mean()
        bl = boundary(net, xl, yl, symmetric)
        br = boundary(net, xr, yr, symmetric)
        bc = (
            (bl["phi_x"] ** 2).mean()
            + (bl["flux_cp"] ** 2).mean()
            + (bl["flux_cn"] ** 2).mean()
            + (br["phi"] ** 2).mean()
            + (br["flux_cp"] ** 2).mean()
            + (br["flux_cn"] ** 2).mean()
        )
        o = net(torch.cat([x0, y0], dim=1))
        a, b, ph = o[:, 0:1], o[:, 1:2], o[:, 2:3]
        cp0, cn0 = (a + b, a - b) if symmetric else (a, b)
        ic = ((cp0 - 1) ** 2).mean() + ((cn0 - 1) ** 2).mean() + (ph**2).mean()
        return {"pde": float(pde), "bc": float(bc), "ic": float(ic)}

    return closure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--activation", default="silu")
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--nr-steps", type=int, default=24)
    ap.add_argument("--range-scale", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-int", type=int, default=16000)
    ap.add_argument("--n-bnd", type=int, default=1600)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = a.device
    net = FullyConnected(act=a.activation).to(dev)
    net.load_physicsnemo(
        torch.load(
            os.path.join(a.run_dir, "net.0.pth"), map_location=dev, weights_only=True
        )
    )
    for p in net.parameters():
        p.requires_grad_(False)

    closure = make_closure(net, a.n_int, a.n_bnd, dev, a.symmetric)
    base = closure()
    print(f"eps={EPS:.6e} xi={XI:g} delta={DELTA:.6f}")
    print(
        "at trained parameters:", {k: f"{v:.4e}" for k, v in base.items()}, flush=True
    )

    ps = list(net.parameters())
    theta = torch.cat([p.data.view(-1) for p in ps]).clone()
    norm = theta.norm().item()
    g = torch.Generator().manual_seed(a.seed)
    u = torch.randn(theta.shape, generator=g).to(dev)
    v = torch.randn(theta.shape, generator=g).to(dev)
    u = u / u.norm() * norm * a.range_scale
    v = v - (v.dot(u) / u.dot(u)) * u
    v = v / v.norm() * norm * a.range_scale

    ax = np.linspace(-1, 1, a.nr_steps)
    surf = {k: np.zeros((a.nr_steps, a.nr_steps)) for k in base}
    for i, al in enumerate(ax):
        for j, be in enumerate(ax):
            nn.utils.vector_to_parameters(theta + al * u + be * v, ps)
            c = closure()
            for k, s in surf.items():
                s[i, j] = c[k]
        print(f"  row {i + 1}/{a.nr_steps}", flush=True)
    nn.utils.vector_to_parameters(theta, ps)

    surf["total"] = sum(surf[k] for k in base)
    out = a.out or os.path.basename(a.run_dir.rstrip("/"))
    np.savez_compressed(out + ".npz", _alphas=ax, _theta_norm=np.array([norm]), **surf)

    rep = {
        "run_dir": a.run_dir,
        "range_scale": a.range_scale,
        "nr_steps": a.nr_steps,
        "seed": a.seed,
        "theta_norm": norm,
        "S": {},
    }
    print("\n  component        L_min        L_max         S")
    for k in list(base) + ["total"]:
        lo, hi = float(surf[k].min()), float(surf[k].max())
        S = hi / lo if lo > 0 else float("inf")
        rep["S"][k] = S
        print(f"  {k:10s} {lo:12.4e} {hi:12.4e} {S:9.3f}")
    with open(out + ".json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {out}.npz / {out}.json")


if __name__ == "__main__":
    main()
