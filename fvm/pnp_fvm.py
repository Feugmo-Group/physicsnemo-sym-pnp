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

"""
FVM solution of PNP system modelling lithium symmetric cell
Reference:
Subramaniam, A., Chen, J., Jang, T., Geise, N. R., Kasse, R. M.,
Toney, M. F., & Subramanian, V. R. (2019). Analysis and Simulation
of One-Dimensional Transport Models for Lithium Symmetric Cells.
Journal of The Electrochemical Society, 166(15), A3806.
doi:10.1149/2.0261915jes
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class Parameters:
    """Simulation parameters"""

    # physical parameters
    c0: float = 500.0  # [mol/m^3] initial electrolyte concentration
    Dp: float = 4.0e-10  # [m^2/s] diffusivity of Li^+
    Dn: float = 4.0e-9  # [m^2/s] diffusivity of PF_6^-
    I_app: float = 10.0  # [A/m^2] applied current density
    L: float = 7.5e-4  # [m] cell dimension in x
    T: float = 298.15  # [K] temperature
    eps0: float = 8.85e-12  # [kg^-1 m^-3 s^4 A^2] vacuum permittivity
    epss: float = 16.8  # [-] dielectric constant of solvent
    R: float = 8.314  # [J K^-1 mol^-1] gas constant
    F: float = 96485.332  # [C/mol] Faraday constant
    zp: int = 1  # [C] cation charge
    zn: int = -1  # [C] anion charge
    t_f: float = 3600  # [s] final time

    # derived, dimensionless parameters
    @property
    def eps(self) -> float:
        return np.sqrt(
            self.R
            * self.T
            * self.epss
            * self.eps0
            / (self.zp**2 * self.F**2 * self.c0 * self.L**2)
        )

    @property
    def xi(self) -> float:
        return self.Dn / self.Dp

    @property
    def delta(self) -> float:
        return self.I_app * self.L / (self.zp * self.F * self.c0 * self.Dp)

    # characteristic time
    @property
    def t_c(self) -> float:
        return self.L**2 / self.Dp


# physical parameters
p = Parameters()
tau_f = p.t_f / p.t_c
t_span = (0.0, tau_f)
eps_sq_inv = 1.0 / (p.eps**2.0)

# simulation parameters
n_elements = 500
n = n_elements - 1
h = 1.0 / n_elements
n_diff_states = 2 * (n + 1)
n_alg_states = n + 1
n_states = n_diff_states + n_alg_states
t_eval = np.linspace(0, tau_f, 3600)

# initial conditions
y0 = np.ones(n_states)
y0[2 * (n + 1) :] = 0.0


def solve_poisson(cp, cn):
    """
    Solve the Poisson equation given concentrations
    """
    # build tridiagonal system for Poisson equation

    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    # boundary condition
    A[0, 0] = 1 / h**2
    A[0, 1] = -1 / h**2
    b[0] = -eps_sq_inv * (cp[0] - cn[0])

    # interior points
    for j in range(1, n):
        A[j, j - 1] = 1 / h**2
        A[j, j] = -2 / h**2
        A[j, j + 1] = 1 / h**2
        b[j] = -eps_sq_inv * (cp[j] - cn[j])

    # boundary condition at n
    A[n, n] = 1.0
    b[n] = 0.0

    # solve linear system
    phi = np.linalg.solve(A, b)
    return phi


def rhs(t, y):
    """
    Compute time derivative for differential ion
    concentrations and Solve algebraic constraint
    to get phi
    """
    cp = y[0 : n + 1]
    cn = y[n + 1 : 2 * (n + 1)]

    # solve Poisson equation to get phi
    phi = solve_poisson(cp, cn)

    # initialize derivatives
    cp_t = np.zeros(n + 1)
    cn_t = np.zeros(n + 1)

    # compute cation fluxes and derivatives
    cp_1 = 0.5 * (cp[0] + cp[1])
    np_1 = -(cp[1] - cp[0]) / h - cp_1 * (phi[1] - phi[0]) / h
    np_0 = p.delta
    cp_t[0] = -(np_1 - np_0) / h

    cp_n = 0.5 * (cp[n - 1] + cp[n])
    np_n = -(cp[n] - cp[n - 1]) / h - cp_n * (phi[n] - phi[n - 1]) / h
    np_npl = p.delta
    cp_t[n] = -(np_npl - np_n) / h

    for j in range(1, n):
        cp_j = 0.5 * (cp[j] + cp[j + 1])
        cp_jml = 0.5 * (cp[j - 1] + cp[j])
        np_j = -(cp[j + 1] - cp[j]) / h - cp_j * (phi[j + 1] - phi[j]) / h
        np_jml = -(cp[j] - cp[j - 1]) / h - cp_jml * (phi[j] - phi[j - 1]) / h
        cp_t[j] = -(np_j - np_jml) / h

    # compute anion fluxes and derivatives
    cn_1 = 0.5 * (cn[0] + cn[1])
    nn_1 = -p.xi * ((cn[1] - cn[0]) / h - cn_1 * (phi[1] - phi[0]) / h)
    nn_0 = 0.0
    cn_t[0] = -(nn_1 - nn_0) / h

    cn_n = 0.5 * (cn[n - 1] + cn[n])
    nn_n = -p.xi * ((cn[n] - cn[n - 1]) / h - cn_n * (phi[n] - phi[n - 1]) / h)
    nn_npl = 0.0
    cn_t[n] = -(nn_npl - nn_n) / h

    for j in range(1, n):
        cn_j = 0.5 * (cn[j] + cn[j + 1])
        cn_jml = 0.5 * (cn[j - 1] + cn[j])
        nn_j = -p.xi * ((cn[j + 1] - cn[j]) / h - cn_j * (phi[j + 1] - phi[j]) / h)
        nn_jml = -p.xi * ((cn[j] - cn[j - 1]) / h - cn_jml * (phi[j] - phi[j - 1]) / h)
        cn_t[j] = -(nn_j - nn_jml) / h

    # return derivatives for differential variables and zeros for algebraic
    return np.concatenate([cp_t, cn_t, phi * 0])


if __name__ == "__main__":
    # solve the system
    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method="Radau",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    print(sol)

    # extract solution
    cp = sol.y[0 : n + 1, :].T
    cn = sol.y[n + 1 : 2 * (n + 1), :].T

    # recompute phi for plotting
    phi = np.array([solve_poisson(cp[i, :], cn[i, :]) for i in range(len(t_eval))])

    x = np.linspace(0, 1, n + 1)
    X, T = np.meshgrid(x, t_eval)

    # store output data
    data = {
        "x": X.flatten(),
        "y": T.flatten(),
        "cp": cp.flatten(),
        "cn": cn.flatten(),
        "phi": phi.flatten(),
    }

    # export to csv
    df = pd.DataFrame(data)
    df.to_csv("./pnp.csv", index=False)
