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


import sympy
from sympy import Symbol, Abs, sqrt, Max, Min
import numpy as np
from typing import Union, Callable
from chaospy.distributions.sampler.sequences.van_der_corput import (
    create_van_der_corput_samples as create_samples,
)

from physicsnemo.sym.geometry.geometry import Geometry, csg_curve_naming
from physicsnemo.sym.geometry.curve import SympyCurve
from physicsnemo.sym.geometry.parameterization import (
    Parameterization,
    Bounds,
    Parameter,
)
from physicsnemo.sym.geometry.helper import (
    _sympy_sdf_to_sdf,
    _sympy_criteria_to_criteria,
)


class GridRectangle(Geometry):
    """2D Rectangle with grid-based sampling"""

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols
        ls = Symbol(csg_curve_naming(0))
        x, y = Symbol("x"), Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({ls: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]

        line_1 = SympyCurve(
            functions={
                "x": ls * dist_x + point_1[0],
                "y": point_1[1],
                "normal_x": 0,
                "normal_y": -1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        line_2 = SympyCurve(
            functions={
                "x": point_2[0],
                "y": ls * dist_y + point_1[1],
                "normal_x": 1,
                "normal_y": 0,
            },
            parameterization=curve_parameterization,
            area=dist_y,
        )
        line_3 = SympyCurve(
            functions={
                "x": ls * dist_x + point_1[0],
                "y": point_2[1],
                "normal_x": 0,
                "normal_y": 1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        line_4 = SympyCurve(
            functions={
                "x": point_1[0],
                "y": -ls * dist_y + point_2[1],
                "normal_x": -1,
                "normal_y": 0,
            },
            parameterization=curve_parameterization,
            area=dist_y,
        )
        curves = [line_1, line_2, line_3, line_4]

        # calculate SDF
        center_x = point_1[0] + (dist_x) / 2
        center_y = point_1[1] + (dist_y) / 2
        x_diff = Abs(x - center_x) - (point_2[0] - center_x)
        y_diff = Abs(y - center_y) - (point_2[1] - center_y)
        outside_distance = sqrt(Max(x_diff, 0) ** 2 + Max(y_diff, 0) ** 2)
        inside_distance = Min(Max(x_diff, y_diff), 0)
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )

    def sample_interior(
        self,
        nr_points: int,
        bounds: Union[Bounds, None] = None,
        criteria: Union[sympy.Basic, None] = None,
        parameterization: Union[Parameterization, None] = None,
        compute_sdf_derivatives: bool = False,
        quasirandom: bool = False,
        flip_interior: bool = False,
    ):
        """Sample interior using structured grid"""
        # compile criteria
        if criteria is not None:
            if isinstance(criteria, sympy.Basic):
                criteria = _sympy_criteria_to_criteria(criteria)
            elif isinstance(criteria, Callable):
                pass
            else:
                raise TypeError("criteria type not supported: " + str(type(criteria)))

        # use internal bounds if not given
        if bounds is None:
            bounds = self.bounds
        elif isinstance(bounds, dict):
            bounds = Bounds(bounds)

        # use internal parameterization if not given
        if parameterization is None:
            parameterization = self.parameterization
        elif isinstance(parameterization, dict):
            parameterization = Parameterization(parameterization)

        # get bounds
        computed_bounds = bounds._compute_bounds(parameterization)
        x_bounds = computed_bounds[Parameter("x")]
        y_bounds = computed_bounds[Parameter("y")]

        # calculate grid dimensions
        aspect_ratio = (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0])
        nx = int(np.sqrt(nr_points * aspect_ratio))
        ny = int(np.sqrt(nr_points / aspect_ratio))
        nx = max(nx, 1)
        ny = max(ny, 1)

        # generate grid points per axis
        if quasirandom:
            x_samples = create_samples(np.arange(nx), number_base=2)
            y_samples = create_samples(np.arange(ny), number_base=3)
            x_grid = x_bounds[0] + x_samples * (x_bounds[1] - x_bounds[0])
            y_grid = y_bounds[0] + y_samples * (y_bounds[1] - y_bounds[0])
        else:
            x_grid = np.linspace(x_bounds[0], x_bounds[1], nx)
            y_grid = np.linspace(y_bounds[0], y_bounds[1], ny)

        # create meshgrid
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

        # flatten
        local_invar = {"x": X.flatten().reshape(-1, 1), "y": Y.flatten().reshape(-1, 1)}

        # sample parameters
        total_grid_points = nx * ny
        local_params = parameterization.sample(total_grid_points, quasirandom=False)

        # evaluate SDF
        local_invar.update(
            self.sdf(
                local_invar,
                local_params,
                compute_sdf_derivatives=compute_sdf_derivatives,
            )
        )

        # filter points
        if flip_interior:
            criteria_index = np.less(local_invar["sdf"], 0)
        else:
            criteria_index = np.greater(local_invar["sdf"], 0)

        if criteria is not None:
            user_criteria = criteria(local_invar, local_params)
            if user_criteria.dtype != bool:
                user_criteria = user_criteria.astype(bool)
            criteria_index = np.logical_and(criteria_index, user_criteria)

        # apply filtering
        for key in local_invar.keys():
            local_invar[key] = local_invar[key][criteria_index[:, 0], :]
        for key in local_params.keys():
            local_params[key] = local_params[key][criteria_index[:, 0], :]

        # compute area
        volume = bounds.volume(parameterization)
        actual_points = next(iter(local_invar.values())).shape[0]

        if actual_points > 0:
            local_invar["area"] = np.full_like(
                next(iter(local_invar.values())), volume / actual_points
            )
        else:
            raise RuntimeError("Could not sample interior. Check non-zero volume")

        # add params
        local_invar.update(local_params)
        return local_invar

    def sample_boundary(
        self,
        nr_points: int,
        criteria: Union[sympy.Basic, None] = None,
        parameterization: Union[Parameterization, None] = None,
        quasirandom: bool = False,
    ):
        """Sample boundary using grid on each edge"""
        # compile criteria
        if criteria is not None:
            if isinstance(criteria, sympy.Basic):
                criteria = _sympy_criteria_to_criteria(criteria)
            elif isinstance(criteria, Callable):
                pass
            else:
                raise TypeError("criteria type not supported: " + str(type(criteria)))

        # use internal parameterization if not given
        if parameterization is None:
            parameterization = self.parameterization
        elif isinstance(parameterization, dict):
            parameterization = Parameterization(parameterization)

        # get bounds
        computed_bounds = self.bounds._compute_bounds(parameterization)
        x_bounds = computed_bounds[Parameter("x")]
        y_bounds = computed_bounds[Parameter("y")]

        # calculate edge lengths
        width = x_bounds[1] - x_bounds[0]
        height = y_bounds[1] - y_bounds[0]
        perimeter = 2 * (width + height)

        # distribute points proportionally
        edge_lengths = [width, height, width, height]
        edge_ratios = [length / perimeter for length in edge_lengths]
        points_per_edge = [max(1, int(nr_points * ratio)) for ratio in edge_ratios]

        # adjust total
        total_assigned = sum(points_per_edge)
        if total_assigned < nr_points:
            longest_idx = np.argmax(edge_lengths)
            points_per_edge[longest_idx] += nr_points - total_assigned

        # sample each edge
        list_invar = []
        list_params = []

        for edge_idx, n_points in enumerate(points_per_edge):
            if n_points > 0:
                # generate t values
                if quasirandom:
                    t_indices = np.arange(n_points)
                    t_values = create_samples(t_indices, number_base=2 + edge_idx)
                else:
                    t_values = np.linspace(0, 1, n_points)

                # get edge parameterization
                edge_param = Parameterization(
                    {
                        self.curves[edge_idx].parameterization.parameters[
                            0
                        ]: t_values.reshape(-1, 1)
                    }
                )
                combined_param = parameterization.union(edge_param)

                # sample from curve's _sample method directly
                invar, params = self.curves[edge_idx]._sample(
                    n_points, combined_param, quasirandom=False
                )

                # set area
                edge_area = edge_lengths[edge_idx]
                invar["area"] = np.full_like(invar["area"], edge_area / n_points)

                # apply criteria if needed
                if criteria is not None:
                    computed_criteria = criteria(invar, params)
                    if computed_criteria.dtype != bool:
                        computed_criteria = computed_criteria.astype(bool)
                    for key in invar.keys():
                        invar[key] = invar[key][computed_criteria[:, 0], :]
                    for key in params.keys():
                        params[key] = params[key][computed_criteria[:, 0], :]

                list_invar.append(invar)
                list_params.append(params)

        # concatenate
        if len(list_invar) == 0:
            raise RuntimeError("Could not sample boundary")

        invar = {
            key: np.concatenate([d[key] for d in list_invar], axis=0)
            for key in list_invar[0].keys()
        }
        params = (
            {
                key: np.concatenate([d[key] for d in list_params], axis=0)
                for key in list_params[0].keys()
            }
            if len(list_params[0]) > 0
            else {}
        )

        invar.update(params)
        return invar
