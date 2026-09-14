"""Loss-landscape scan and loss variation ratio S = L_max / L_min for the 1D PNP benchmark.

Reloads a trained PhysicsNeMo Sym checkpoint, sweeps the training loss over a
two-dimensional plane of parameter perturbations, writes the raw surfaces to .npz, and
reports S over the swept region.

Unlike the generic landscape helpers, the closure here keeps autograd enabled: the PNP
residuals need second spatial derivatives, so the loss cannot be evaluated under
torch.no_grad(). Parameter perturbations are applied in-place and restored afterwards.

Usage (inside the physicsnemo container, from the repository root):

    python landscape_sharpness.py --checkpoint outputs/pnp_ntk/net.0.pth \
        --arch fully_connected --nr-steps 24 --range-scale 0.5 --out landscape_ntk

Outputs <out>.npz (per-component surfaces + axes) and prints S for each component and
for the total.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.fully_connected import FullyConnectedArch

from pnp import BoundaryConditions, PoissonNernstPlanck
from registry import Parameters


# --------------------------------------------------------------------------- sampling
def halton(n, dim, skip=20):
    """Deterministic Halton sequence, so every configuration sees the same points."""
    primes = [2, 3, 5, 7, 11, 13][:dim]
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


def make_points(n_int, n_bnd, y_f, device):
    """Interior, initial, left and right collocation points; fixed for the whole scan."""
    p = halton(n_int, 2)
    interior = {
        "x": torch.tensor(
            p[:, 0:1], dtype=torch.float32, device=device, requires_grad=True
        ),
        "y": torch.tensor(
            p[:, 1:2] * y_f, dtype=torch.float32, device=device, requires_grad=True
        ),
    }
    q = halton(n_bnd, 1, skip=37)
    initial = {
        "x": torch.tensor(q, dtype=torch.float32, device=device, requires_grad=True),
        "y": torch.zeros(
            (n_bnd, 1), dtype=torch.float32, device=device, requires_grad=True
        ),
    }
    left = {
        "x": torch.zeros(
            (n_bnd, 1), dtype=torch.float32, device=device, requires_grad=True
        ),
        "y": torch.tensor(
            q * y_f, dtype=torch.float32, device=device, requires_grad=True
        ),
    }
    right = {
        "x": torch.ones(
            (n_bnd, 1), dtype=torch.float32, device=device, requires_grad=True
        ),
        "y": torch.tensor(
            q * y_f, dtype=torch.float32, device=device, requires_grad=True
        ),
    }
    return interior, initial, left, right


# --------------------------------------------------------------------------- loss closure
def build_closure(net, params, pts, device):
    """Return a callable giving {component: mse} for the current network parameters."""
    pnp = PoissonNernstPlanck(eps=params.eps, xi=params.xi)
    bc = BoundaryConditions(delta=params.delta)
    node = net.make_node(name="net")
    nodes = pnp.make_nodes() + bc.make_nodes() + [node]

    inputs = [Key("x"), Key("y")]
    g_int = Graph(
        nodes, inputs, [Key("poisson"), Key("continuity_p"), Key("continuity_n")]
    )
    g_left = Graph(
        nodes,
        inputs,
        [Key("neumann_phi_left"), Key("flux_cp_left"), Key("flux_cn_left")],
    )
    g_right = Graph(
        nodes,
        inputs,
        [Key("dirichlet_phi_right"), Key("flux_cp_right"), Key("flux_cn_right")],
    )
    g_ic = Graph(nodes, inputs, [Key("cp"), Key("cn"), Key("phi")])
    for g in (g_int, g_left, g_right, g_ic):
        g.to(device)

    interior, initial, left, right = pts

    def closure():
        out = {}
        r = g_int(interior)
        out["pde"] = float(
            sum(
                torch.mean(r[k] ** 2)
                for k in ("poisson", "continuity_p", "continuity_n")
            ).detach()
        )
        rl = g_left(left)
        rr = g_right(right)
        out["bc"] = float(
            (
                sum(torch.mean(rl[k] ** 2) for k in rl)
                + sum(torch.mean(rr[k] ** 2) for k in rr)
            ).detach()
        )
        ic = g_ic(initial)
        out["ic"] = float(
            (
                torch.mean((ic["cp"] - 1.0) ** 2)
                + torch.mean((ic["cn"] - 1.0) ** 2)
                + torch.mean(ic["phi"] ** 2)
            ).detach()
        )
        return out

    return closure


# --------------------------------------------------------------------------- scan
def scan(net, closure, nr_steps, range_scale, seed):
    ps = list(net.parameters())
    theta = torch.cat([p.data.view(-1) for p in ps]).clone()
    norm = theta.norm().item() + 1e-12

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    d1 = torch.randn(theta.shape, generator=rng, dtype=theta.dtype).to(theta.device)
    d2 = torch.randn(theta.shape, generator=rng, dtype=theta.dtype).to(theta.device)
    d1 = (d1 / d1.norm()) * norm * range_scale
    d2 = d2 - (d2.dot(d1) / (d1.dot(d1) + 1e-12)) * d1
    d2 = (d2 / (d2.norm() + 1e-12)) * norm * range_scale

    alphas = np.linspace(-1.0, 1.0, nr_steps)
    betas = np.linspace(-1.0, 1.0, nr_steps)

    probe = closure()
    surfs = {k: np.zeros((nr_steps, nr_steps)) for k in probe}
    n = nr_steps * nr_steps
    done = 0
    for i, a in enumerate(alphas):
        ad = a * d1
        for j, b in enumerate(betas):
            nn.utils.vector_to_parameters(theta + ad + b * d2, ps)
            c = closure()
            for k, surf in surfs.items():
                surf[i, j] = c[k]
            done += 1
            if done % max(1, n // 10) == 0:
                print(f"  {100 * done // n:3d}%", flush=True)
    nn.utils.vector_to_parameters(theta, ps)

    surfs["total"] = sum(surfs[k] for k in probe)
    surfs["_alphas"] = alphas
    surfs["_betas"] = betas
    surfs["_theta_norm"] = np.array([norm])
    return surfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--arch", default="fully_connected", choices=["fully_connected", "fourier"]
    )
    ap.add_argument("--layer-size", type=int, default=512)
    ap.add_argument("--nr-layers", type=int, default=6)
    ap.add_argument("--activation", default="silu")
    ap.add_argument("--nr-steps", type=int, default=24)
    ap.add_argument("--range-scale", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-int", type=int, default=16000)
    ap.add_argument("--n-bnd", type=int, default=1600)
    ap.add_argument("--out", default="landscape")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = Parameters()
    print(f"eps={params.eps:.6e}  xi={params.xi}  delta={params.delta:.6f}", flush=True)

    from physicsnemo.sym.models.activation import Activation

    cls = FullyConnectedArch if args.arch == "fully_connected" else FourierNetArch
    net = cls(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("cp"), Key("cn"), Key("phi")],
        layer_size=args.layer_size,
        nr_layers=args.nr_layers,
        activation_fn=Activation[args.activation.upper()],
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(state)
    net.eval()
    print(f"loaded {args.checkpoint}", flush=True)

    pts = make_points(args.n_int, args.n_bnd, params.t_f / params.t_c, device)
    closure = build_closure(net, params, pts, device)

    at_min = closure()
    print(
        "loss at the trained parameters:",
        {k: f"{v:.4e}" for k, v in at_min.items()},
        flush=True,
    )

    surfs = scan(net, closure, args.nr_steps, args.range_scale, args.seed)

    np.savez_compressed(args.out + ".npz", **surfs)
    report = {
        "checkpoint": args.checkpoint,
        "nr_steps": args.nr_steps,
        "range_scale": args.range_scale,
        "seed": args.seed,
        "theta_norm": float(surfs["_theta_norm"][0]),
        "S": {},
    }
    print("\n  component        L_min        L_max            S")
    for k in [c for c in surfs if not c.startswith("_")]:
        s = surfs[k]
        lo, hi = float(s.min()), float(s.max())
        S = hi / lo if lo > 0 else float("inf")
        report["S"][k] = S
        print(f"  {k:10s} {lo:12.4e} {hi:12.4e} {S:12.3f}")
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}.npz and {args.out}.json")


if __name__ == "__main__":
    main()
