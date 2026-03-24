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


from dataclasses import dataclass
import numpy as np


@dataclass
class Parameters:
    """Simulation parameters"""

    # physical parameters
    c0: float = 500.0  # [mol/m^3] initial electrolyte concentration
    Dp: float = 4.0e-10  # [m^2/s] diffusivity of Li^+
    Dn: float = 4.0e-9  # [m^2/s] diffusivity of PF_6^-
    I_app: float = 10.0  # [A^2/m] applied current density
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
