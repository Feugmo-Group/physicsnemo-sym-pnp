# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import warnings
import numpy as np
from sympy import Symbol, Function, Number, Eq
from matplotlib import colormaps, use
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from typing import Tuple
from registry import (
    register_custom_arch_configs,
    register_custom_loss_configs,
    Parameters,
    GridRectangle,
)

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
)

use("Agg")


class PoissonNernstPlanck(PDE):
    """
    Dimensionless 1D Poisson-Nernst-Planck (PNP) system for two ionic species
    Reference:
    Subramaniam, A., Chen, J., Jang, T., Geise, N. R., Kasse, R. M.,
    Toney, M. F., & Subramanian, V. R. (2019). Analysis and Simulation
    of One-Dimensional Transport Models for Lithium Symmetric Cells.
    Journal of The Electrochemical Society, 166(15), A3806.
    doi:10.1149/2.0261915jes

    Parameters
    ==========
    eps : float, optional
        Dimensionless Poisson parameter. Defined as
        sqrt(R * T * eps_s * eps_0 / (z_p ** 2 * F ** 2 * c_0 * L ** 2))
        Default is 1.
    xi : float, optional
        Dimensionless ratio of diffusion coefficients: D_p / D_n. Default is 1.

    Example
    ========
    >>> pnp = PoissonNernstPlanck(eps=0.1, xi=0.1)
    >>> pnp.pprint()
    poisson: -cn + cp + 0.01*phi__x__x
    continuity_p: -cp*phi__x__x - cp__x*phi__x - cp__x__x + cp__y
    continuity_n: 0.1*cn*phi__x__x + 0.1*cn__x*phi__x - 0.1*cn__x__x + cn__y
    """

    name = "PoissonNernstPlanck"

    def __init__(self, eps=1.0, xi=1.0):
        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # make cp, cn, and phi functions
        cp = Function("cp")(*input_variables)
        cn = Function("cn")(*input_variables)
        phi = Function("phi")(*input_variables)

        # nondimensional constants
        eps = Number(eps)
        xi = Number(xi)

        # set equations
        self.equations = {}
        self.equations["poisson"] = eps**2 * phi.diff(x, 2) + (cp - cn)
        self.equations["continuity_p"] = cp.diff(y, 1) - (
            cp.diff(x, 2) + cp * phi.diff(x, 2) + cp.diff(x, 1) * phi.diff(x, 1)
        )
        self.equations["continuity_n"] = cn.diff(y, 1) - xi * (
            cn.diff(x, 2) - cn * phi.diff(x, 2) - cn.diff(x, 1) * phi.diff(x, 1)
        )


class BoundaryConditions(PDE):
    """
    Boundary conditions for lithium symmetric cell 1D PNP system
    Reference:
    Subramaniam, A., Chen, J., Jang, T., Geise, N. R., Kasse, R. M.,
    Toney, M. F., & Subramanian, V. R. (2019). Analysis and Simulation
    of One-Dimensional Transport Models for Lithium Symmetric Cells.
    Journal of The Electrochemical Society, 166(15), A3806.
    doi:10.1149/2.0261915jes

    Parameters
    ==========
    delta : float, optional
        Dimensionless cation flux parameter. Defined as
        I_app * L / (z_p * F * c_0 * D_p)
        Default is 1.

    Example
    ========
    >>> bc = BoundaryConditions(delta=0.1)
    >>> bc.pprint()
    neumann_phi_left: phi__x
    flux_cp_left: -cp*phi__x - cp__x - 0.1
    flux_cn_left: cn*phi__x - cn__x
    dirichlet_phi_right: phi
    flux_cp_right: -cp*phi__x - cp__x - 0.1
    flux_cn_right: cn*phi__x - cn__x
    """

    name = "BoundaryConditions"

    def __init__(self, delta=1.0):
        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # make cp, cn, and phi functions
        cp = Function("cp")(*input_variables)
        cn = Function("cn")(*input_variables)
        phi = Function("phi")(*input_variables)

        # nondimensional constants
        delta = Number(delta)

        self.equations = {}

        # left boundary (x=0)
        self.equations["neumann_phi_left"] = phi.diff(x, 1)
        self.equations["flux_cp_left"] = -cp.diff(x, 1) - cp * phi.diff(x, 1) - delta
        self.equations["flux_cn_left"] = -cn.diff(x, 1) + cn * phi.diff(x, 1)

        # right boundary (x=1)
        self.equations["dirichlet_phi_right"] = phi
        self.equations["flux_cp_right"] = -cp.diff(x, 1) - cp * phi.diff(x, 1) - delta
        self.equations["flux_cn_right"] = -cn.diff(x, 1) + cn * phi.diff(x, 1)


class PNPValidatorPlotter(ValidatorPlotter):
    """
    Plotter class for validating PNP space-time solutions and spatial profiles

    Parameters
    ==========
    times : Tuple[float], optional
        Times plot spatial profiles at, by default (0.0, 1.0, 6.0, 36.0, 100.0, 3600.0)
    t_c : float, optional
        Characteristic time, by default 1.0
    """

    def __init__(
        self,
        times: Tuple[float] = (0.0, 1.0, 6.0, 36.0, 100.0, 3600.0),
        t_c: float = 1.0,
    ):
        super().__init__()
        self.t_c = t_c
        self.times = times
        self._darken = lambda col: tuple(c * 0.7 for c in col[:3])

        # store colors from color map
        cmap = colormaps["Dark2"]
        self._colors = cmap(np.linspace(0, 0.6, len(self.times)))
        self._darker_colors = [tuple(c * 0.7 for c in col[:3]) for col in self._colors]

        # create cache for interpolated true variables
        self.true_outvar = None

    def _add_figures(self, group, name, results_dir, writer, step, *args):
        """Try to make plots and write them to tensorboard summary"""

        # catch exceptions on (possibly user-defined) __call__
        try:
            fs = self(*args)
        except Exception as e:
            print(f"error: {self}.__call__ raised an exception:", str(e))
        else:
            for f, tag in fs:
                f.savefig(
                    results_dir + name + "_" + tag + "_" + str(step) + "_epochs.png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                writer.add_figure(group + "/" + name + "/" + tag, f, step, close=True)
            plt.close("all")

    def __call__(self, invar, true_outvar, pred_outvar):
        """
        Plot true and predicted space-time solutions, their difference,
        and spatial profiles for each predicted field
        """
        # make spatial profiles
        # create figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

        axs[0].set_ylabel("cp")
        axs[1].set_ylabel("cn")
        axs[2].set_ylabel("phi")

        for ax in axs:
            ax.set_xlabel("x")
            ax.set_box_aspect(1)

        x_raw = invar["x"]
        y_raw = invar["y"]

        cp_pred_raw = pred_outvar["cp"]
        cn_pred_raw = pred_outvar["cn"]
        phi_pred_raw = pred_outvar["phi"]

        cp_true_raw = true_outvar["cp"]
        cn_true_raw = true_outvar["cn"]
        phi_true_raw = true_outvar["phi"]

        # group plots by time
        for i, t in enumerate(self.times):
            target_y = t / self.t_c

            # find indices where y matches the target time
            mask = np.isclose(y_raw, target_y, atol=1e-5).flatten()

            if np.any(mask):
                # get colors
                col = self._colors[i]
                col_dark = self._darker_colors[i]

                # sort by x to ensure line plots look correct
                sort_idx = np.argsort(x_raw[mask].flatten())
                xs = x_raw[mask][sort_idx]

                # predictions are solid
                axs[0].plot(
                    xs,
                    cp_pred_raw[mask][sort_idx],
                    label=f"t={t:.1f}s",
                    color=col,
                    linestyle="-",
                    alpha=0.9,
                )
                axs[1].plot(
                    xs,
                    cn_pred_raw[mask][sort_idx],
                    label=f"t={t:.1f}s",
                    color=col,
                    linestyle="-",
                    alpha=0.9,
                )
                axs[2].plot(
                    xs,
                    phi_pred_raw[mask][sort_idx],
                    label=f"t={t:.1f}s",
                    color=col,
                    linestyle="-",
                    alpha=0.9,
                )

                # true profiles are dotted
                axs[0].plot(
                    xs,
                    cp_true_raw[mask][sort_idx],
                    label=f"t={t:.1f}s",
                    color=col_dark,
                    linestyle=":",
                    zorder=3,
                    alpha=0.9,
                )
                axs[1].plot(
                    xs,
                    cn_true_raw[mask][sort_idx],
                    label=f"t={t:.1f}s",
                    color=col_dark,
                    linestyle=":",
                    zorder=3,
                    alpha=0.9,
                )
                axs[2].plot(
                    xs,
                    phi_true_raw[mask][sort_idx],
                    label=f"t={t:.1f}s",
                    color=col_dark,
                    linestyle=":",
                    zorder=3,
                    alpha=0.9,
                )

        for ax in axs:
            ax.legend(framealpha=0.0)

        # add legend for true versus pred
        dashed_proxy = Line2D([0], [0], color="gray", linestyle=":", label="True")
        solid_proxy = Line2D([0], [0], color="gray", linestyle="-", label="Prediction")
        fig.legend(
            handles=[dashed_proxy, solid_proxy],
            loc="upper center",
            ncol=2,
            framealpha=0.0,
        )

        plt.tight_layout()

        figures = [(fig, "spatial_profiles")]

        # make space-time validation plots
        # interpolate
        if self.true_outvar is None:
            extent, true_outvar, pred_outvar = self._interpolate_2D(
                100, invar, true_outvar, pred_outvar
            )
            self.true_outvar = true_outvar
        else:
            true_outvar = self.true_outvar
            extent, pred_outvar = self._interpolate_2D(100, invar, pred_outvar)

        # make one validation plot for each field
        for key in pred_outvar:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

            true_out = true_outvar[key]
            pred_out = pred_outvar[key]
            diff_out = true_out - pred_out
            tags = ["True", "Predicted", "Difference"]

            # make subplots
            for i, (out, tag) in enumerate(zip([true_out, pred_out, diff_out], tags)):
                axs[i].set_box_aspect(1)
                axs[i].set_xlabel("x")
                axs[i].set_ylabel("tau")
                axs[i].set_title(f"{tag} {key}")
                contour = axs[i].imshow(
                    out.T,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                )
                fig.colorbar(contour)

            plt.tight_layout()
            figures.append((fig, key))

        return figures


@physicsnemo.sym.main(config_path="conf", config_name="config_kan")
def run(cfg: PhysicsNeMoConfig) -> None:
    # instantiate simulation parameters
    p = Parameters()

    # make a list of nodes for the graph to unroll on
    pnp = PoissonNernstPlanck(eps=p.eps, xi=p.xi)
    bc = BoundaryConditions(delta=p.delta)
    arch_cfg = cfg.arch[next(iter(cfg.arch))]

    # make neural nets for each field
    # cation concentration
    cp_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("cp")],
        cfg=arch_cfg,
    )

    # anion concentration
    cn_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("cn")],
        cfg=arch_cfg,
    )

    # electric potential
    phi_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("phi")],
        cfg=arch_cfg,
    )

    nodes = (
        pnp.make_nodes()
        + bc.make_nodes()
        + [cp_net.make_node(name="cp_net")]
        + [cn_net.make_node(name="cn_net")]
        + [phi_net.make_node(name="phi_net")]
    )

    # add constraints to solver
    # make geometry
    x, y = Symbol("x"), Symbol("y")
    y_f = p.t_f / p.t_c  # final dimensionless time

    if cfg.custom.grid_sampling:
        rec = GridRectangle((0.0, 0.0), (1.0, y_f))
    else:
        rec = Rectangle((0.0, 0.0), (1.0, y_f))

    # make pnp domain
    pnp_domain = Domain()

    quasirandom = cfg.custom.quasirandom

    # initial condition
    initial = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"cp": 1.0, "cn": 1.0, "phi": 0.0},
        batch_size=cfg.batch_size.Initial,
        criteria=Eq(y, 0.0),
        quasirandom=quasirandom,
        fixed_dataset=False,
    )
    pnp_domain.add_constraint(initial, "initial")

    # left boundary
    left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"flux_cn_left": 0.0, "flux_cp_left": 0.0, "neumann_phi_left": 0.0},
        batch_size=cfg.batch_size.Left,
        criteria=Eq(x, 0.0),
        quasirandom=quasirandom,
        fixed_dataset=False,
    )
    pnp_domain.add_constraint(left, "left")

    # right boundary
    right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"flux_cn_right": 0.0, "flux_cp_right": 0.0, "dirichlet_phi_right": 0.0},
        batch_size=cfg.batch_size.Right,
        criteria=Eq(x, 1.0),
        quasirandom=quasirandom,
        fixed_dataset=False,
    )
    pnp_domain.add_constraint(right, "right")

    # interior
    lambda_weighting = None
    if cfg.custom.sdf:
        lambda_weighting = {
            "poisson": Symbol("sdf"),
            "continuity_p": Symbol("sdf"),
            "continuity_n": Symbol("sdf"),
        }
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"poisson": 0.0, "continuity_p": 0.0, "continuity_n": 0.0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting=lambda_weighting,
        quasirandom=quasirandom,
        fixed_dataset=False,
    )
    pnp_domain.add_constraint(interior, "interior")

    # add validator
    file_path = "fvm/pnp.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"x": "x", "y": "y", "cp": "cp", "cn": "cn", "phi": "phi"}
        fvm_var = csv_to_dict(to_absolute_path(file_path), mapping)
        fvm_invar_numpy = {
            key: value for key, value in fvm_var.items() if key in ["x", "y"]
        }
        fvm_outvar_numpy = {
            key: value for key, value in fvm_var.items() if key in ["cp", "cn", "phi"]
        }
        fvm_validator = PointwiseValidator(
            nodes=nodes,
            invar=fvm_invar_numpy,
            true_outvar=fvm_outvar_numpy,
            batch_size=1024,
            plotter=PNPValidatorPlotter(t_c=p.t_c),
        )
        pnp_domain.add_validator(fvm_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators."
        )

    # make solver
    slv = Solver(cfg, pnp_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    register_custom_arch_configs()
    register_custom_loss_configs()

    run()
