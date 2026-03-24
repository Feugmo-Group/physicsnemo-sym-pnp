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


from dataclasses import dataclass, field

from physicsnemo.sym.models.utils import register_arch, PhysicsNeMoModels
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from physicsnemo.sym.key import Key
from physicsnemo.sym.models.activation import Activation, get_activation_fn
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.hydra.arch import ModelConf
from hydra.core.config_store import ConfigStore


class FiniteBasisNetArch(Arch):
    """
    Finite basis neural network based on overlapping domain
    decomposition
    Reference:
    Dolean, V., Heinlein, A., Mishra, S., & Moseley, B. (2024).
    Multilevel domain decomposition-based architectures for
    physics-informed neural networks. Computer Methods in Applied
    Mechanics and Engineering, 429, 117116.
    doi:10.1016/j.cma.2024.117116

    Parameters
    ----------
    input_keys : List[Key]
        Input key list.
    output_keys : List[Key]
        Output key list.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int, optional
        Layer size for every hidden layer of each subnetwork, by default 512
    nr_layers : int, optional
        Number of hidden layers of each subnetwork, by default 6
    nr_levels : int, optional
        Number of domain decomposition levels, be default 4
    refinement_factor : int, optional
        Each level increases the number of subdomains by this factor starting from one, by default 2
    reduction_factor : int, optional
        Each level decreases the layer size of subnetworks by this factor, by default 1
    overlap_ratio : float, optional
        Overlap ratio between subdomains (between 1.0 and 3.0), by default 2.7
    window_fn : str, optional
        Type of window function ('cosine', 'sigmoid', 'bump'), by default 'cosine'
    subnet_arch_type : str, optional
        Name of architecture class to use for subnetworks, by default 'fully_connected'
    subnet_kwargs : dict, optional
        Additional keyword arguments to pass to subnet architecture, by default {}
    domain_bounds : Tuple[Tuple[float, float], ...], optional
        Domain bounds for each dimension as ((x_min, x_max), (y_min, y_max), ...),
        assumes [0, 1] if none, by default None.
    activation_fn : Activation, optional
        Activation function used by network, by default :obj:`Activation.SILU`
    skip_connections : bool, optional
        Apply skip connections in subnetworks, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default True
    adaptive_activations : bool, optional
        Use adaptive activation functions, by default False
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        nr_levels: int = 4,
        refinement_factor: int = 2,
        reduction_factor: int = 1,
        overlap_ratio: float = 2.7,
        window_fn: str = "cosine",
        subnet_arch_type: str = "fully_connected",
        subnet_kwargs: dict = {},
        domain_bounds: Optional[List[Any]] = None,
        activation_fn: Activation = Activation.SILU,
        skip_connections: bool = False,
        weight_norm: bool = True,
        adaptive_activations: bool = False,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
        )
        self.nr_levels = nr_levels
        self.overlap_ratio = overlap_ratio
        self.window_fn = window_fn

        # calculate input and output dimension
        self.input_dim = len(input_keys)
        self.output_dim = sum(self.output_key_dict.values())

        # set domain bounds; default to [0,1]
        if domain_bounds is None:
            self.domain_bounds = tuple((0.0, 1.0) for _ in range(self.input_dim))
        else:
            if len(domain_bounds) != self.input_dim:
                raise ValueError(
                    f"FiniteBasisNetArch expects {self.input_dim} domain bounds, "
                    f"got {len(domain_bounds)}"
                )
            domain_bounds = tuple(tuple(float(v) for v in b) for b in domain_bounds)
            self.domain_bounds = domain_bounds

        # decompose domain for each level
        self.subdomains_per_level = torch.zeros(nr_levels, dtype=torch.int)
        self.subdomain_shapes = []

        for level_idx in range(nr_levels):
            subdomain_shape = torch.tensor(
                [refinement_factor**level_idx for _ in range(self.input_dim)]
            )
            self.subdomain_shapes.append(subdomain_shape)

            nr_subdomains = torch.prod(subdomain_shape).item()
            self.subdomains_per_level[level_idx] = nr_subdomains

        self.subdomain_shapes = torch.stack(self.subdomain_shapes)
        self.total_subdomains = self.subdomains_per_level.sum()
        self.subdomains_per_level = self.subdomains_per_level.tolist()

        # get subnet arch class
        pn_models = PhysicsNeMoModels()
        if subnet_arch_type not in pn_models:
            raise ValueError(f"Architecture '{subnet_arch_type}' not found in registry")
        subnet_arch = pn_models[subnet_arch_type]

        # create subdomain networks
        self.subnetworks = nn.ModuleList()
        for level_idx in range(nr_levels):
            level_networks = nn.ModuleList()
            nr_subdomains = self.subdomains_per_level[level_idx]
            subnet_layer_size = max(1, int(layer_size / (reduction_factor**level_idx)))
            for subdomain_idx in range(nr_subdomains):
                subnet = subnet_arch(
                    input_keys=input_keys,
                    output_keys=output_keys,
                    detach_keys=[],
                    layer_size=subnet_layer_size,
                    nr_layers=nr_layers,
                    activation_fn=activation_fn,
                    skip_connections=skip_connections,
                    weight_norm=weight_norm,
                    adaptive_activations=adaptive_activations,
                    **subnet_kwargs,
                )
                level_networks.append(subnet)
            self.subnetworks.append(level_networks)

        # store subdomain parameters
        self._initialize_subdomain_params()

    def _initialize_subdomain_params(self):
        """Initialize subdomain centers and widths for normalization and windows"""
        self.subdomain_centers = []
        self.subdomain_widths = []

        for level_idx in range(self.nr_levels):
            subdomain_shape = self.subdomain_shapes[level_idx]
            nr_dims = len(subdomain_shape)

            # create coordinate vectors for each dimension
            coords = []
            widths = []
            for dim_idx, nr_subdomains in enumerate(subdomain_shape):
                x_min, x_max = self.domain_bounds[dim_idx]
                domain_size = x_max - x_min

                if nr_subdomains == 1:
                    c = torch.tensor([x_min + domain_size * 0.5])
                    w = torch.tensor([domain_size])
                else:
                    # linearly spaced centers across the domain
                    c = torch.linspace(x_min, x_max, nr_subdomains)

                    # constant width based on overlap
                    w = torch.full(
                        (nr_subdomains,),
                        (self.overlap_ratio / (nr_subdomains - 1)) * domain_size,
                    )

                coords.append(c)
                widths.append(w)

            # generate all combinations
            grid_c = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=0)
            grid_w = torch.stack(torch.meshgrid(*widths, indexing="ij"), dim=0)

            # reshape to [total_subdomains, dims]
            level_centers = grid_c.reshape(nr_dims, -1).T
            level_widths = grid_w.reshape(nr_dims, -1).T

            # ensure move to GPU
            self.register_buffer(f"level_{level_idx}_centers", level_centers)
            self.register_buffer(f"level_{level_idx}_widths", level_widths)

            self.subdomain_centers.append(level_centers)
            self.subdomain_widths.append(level_widths)

    def _apply_window(self, x: Tensor, center: Tensor, width: Tensor) -> Tensor:
        """Apply window function to confine subnetwork output"""
        # initialize window
        window = torch.ones([], device=x.device)

        # apply 1D window along each dimension
        for dim_idx in range(self.input_dim):
            x_coord = x[..., dim_idx]
            c = center[dim_idx]
            w = width[dim_idx]
            window_1d = None

            if self.window_fn == "cosine" or self.window_fn is None:
                normalized = (x_coord - c) / (w * 0.5)
                window_1d = ((1 + torch.cos(torch.pi * normalized)) * 0.5) ** 2
                mask = ((x_coord >= (c - w * 0.5)) & (x_coord <= (c + w * 0.5))).float()
                window_1d = mask * window_1d

            elif self.window_fn == "sigmoid":
                sd = (w * 0.5) / 8
                window_1d = torch.sigmoid(
                    (x_coord - (c - w * 0.5)) / sd
                ) * torch.sigmoid(((c + w * 0.5) - x_coord) / sd)

            elif self.window_fn == "bump":
                r_sq = ((x_coord - c) / (w * 0.5)) ** 2
                window_1d = torch.where(
                    r_sq < 1,
                    torch.exp(3 / (r_sq - 1.001)) / 4.9787e-2,
                    torch.zeros_like(r_sq),
                )

            window = window * window_1d

        return window.unsqueeze(-1)

    def _tensor_forward(self, x: Tensor) -> Tensor:
        output = None

        # average contributions from all levels
        for level_idx in range(self.nr_levels):
            output_level = None
            window_total = None

            # access buffers
            centers = getattr(self, f"level_{level_idx}_centers")
            widths = getattr(self, f"level_{level_idx}_widths")

            for sub_idx in range(self.subdomains_per_level[level_idx]):
                c = centers[sub_idx]
                w = widths[sub_idx]

                # apply norm and window
                x_norm = (x - c) / w
                window = self._apply_window(x, c, w)

                # compute subnet forward pass
                subnet = self.subnetworks[level_idx][sub_idx]
                output_subnet = subnet._tensor_forward(x_norm)

                # ensure output is flattened properly for weighting
                output_subnet = output_subnet.flatten(start_dim=-1)
                weighted_output = window * output_subnet

                # accumulate
                if output_level is None:
                    output_level = weighted_output
                    window_total = window
                else:
                    output_level = output_level + weighted_output
                    window_total = window_total + window

            # partition of unity normalization
            norm_level_out = output_level / (window_total + 1e-8)

            if output is None:
                output = norm_level_out
            else:
                output = output + norm_level_out

        return output / self.nr_levels

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)


@dataclass
class FiniteBasisNetConf(ModelConf):
    arch_type: str = "fbpinn"
    layer_size: int = 512
    nr_layers: int = 6
    nr_levels: int = 4
    refinement_factor: int = 2
    reduction_factor: int = 1
    overlap_ratio: float = 2.7
    window_fn: str = "cosine"
    subnet_arch_type: str = "fully_connected"
    subnet_kwargs: dict = field(default_factory=dict)
    domain_bounds: Optional[List[Any]] = None
    activation_fn: str = "silu"
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False


class KANLayer(nn.Module):
    """
    Single Kolmogorov-Arnold Network layer with B-spline activations.
    Reference:
    Wang, Y., Sun, J., Bai, J., Anitescu, C., Eshaghi, M. S.,
    Zhuang, X., … Liu, Y. (2025). Kolmogorov–Arnold-Informed
    neural network: A physics-informed deep learning framework
    for solving forward and inverse problems based on
    Kolmogorov–Arnold Networks. Computer Methods in Applied
    Mechanics and Engineering, 433, 117518.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    grid_size : int, optional
        Number of grid points for B-spline interpolation, by default 100
    spline_order : int, optional
        Order of B-splines, by default 3
    base_activation_fn : nn.Module, optional
        Base activation function for residual connection, by default None
    grid_range : Tuple[float, float], optional
        Range for grid initialization, by default (-1, 1)
    free_knot : bool, optional
        Apply learnable biases to grid for non-uniformity, by default False
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 100,
        spline_order: int = 3,
        base_activation_fn: Optional[nn.Module] = None,
        grid_range: Tuple[float, float] = (-1, 1),
        free_knot: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.free_knot = free_knot

        # base activation function for residual connection
        if base_activation_fn is None or isinstance(base_activation_fn, str):
            self.base_activation_fn = nn.Tanh()
        else:
            self.base_activation_fn = base_activation_fn

        # learnable parameters
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        # initialize grid
        if free_knot:
            self.knot_gaps = nn.Parameter(
                torch.zeros(in_features, grid_size + 2 * spline_order)
            )
            self.register_buffer("_grid_range", torch.tensor(grid_range))
        else:
            h = (grid_range[1] - grid_range[0]) / grid_size
            grid_uniform = (
                torch.linspace(
                    grid_range[0] - h * spline_order,
                    grid_range[1] + h * spline_order,
                    grid_size + 2 * spline_order + 1,
                )
                .expand(in_features, -1)
                .contiguous()
            )
            self.register_buffer("_grid_uniform", grid_uniform, persistent=False)

        # initialize weights
        self.reset_parameters()

    @property
    def grid(self) -> Tensor:
        """
        Implements Free-Knot KAN with learnable bias on grid
        Reference:
        Zheng, L. N., Zhang, W. E., Yue, L., Xu, M., Maennel,
        O., & Chen, W. (2025). Free-Knots Kolmogorov-Arnold
        Network: On the Analysis of Spline Knots and
        Advancing Stability. arXiv [Cs.LG]. Retrieved from
        http://arxiv.org/abs/2501.09283
        """
        if self.free_knot:
            # ensure gaps are strictly positive
            gaps = F.softplus(self.knot_gaps) + 1e-6

            # normalize gaps
            total_width = self._grid_range[1] - self._grid_range[0]
            h_avg = total_width / self.grid_size
            total_padded_width = total_width + (2 * self.spline_order * h_avg)
            normalized_gaps = gaps * (
                total_padded_width / gaps.sum(dim=-1, keepdim=True)
            )

            # build grid via cumulative sum
            start_point = self._grid_range[0] - (self.spline_order * h_avg)
            grid = torch.cumsum(normalized_gaps, dim=-1)

            # prepend zero and offset to start_point
            zeros = torch.zeros(
                self.in_features, 1, device=grid.device, dtype=grid.dtype
            )
            grid = torch.cat([zeros, grid], dim=-1) + start_point

            return grid

        else:
            return self._grid_uniform

    def reset_parameters(self):
        """Initialize weights using Kaiming uniform distribution"""
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity="linear")

    def b_splines(self, x: Tensor) -> Tensor:
        """
        Compute B-spline basis functions
        """
        x = x.to(self.grid.device)
        x_unsqueezed = x.unsqueeze(-1)

        # compute initial basis
        bases = (
            (x_unsqueezed >= self.grid[..., :-1]) & (x_unsqueezed < self.grid[..., 1:])
        ).to(x.dtype)

        # recursively compute higher order B-splines
        for k in range(1, self.spline_order + 1):
            left_intervals = self.grid[..., : -(k + 1)]
            right_intervals = self.grid[..., k:-1]
            next_intervals = self.grid[..., k + 1 :]
            shifted_intervals = self.grid[..., 1:-k]

            delta_left = torch.where(
                right_intervals == left_intervals,
                torch.ones_like(right_intervals),
                right_intervals - left_intervals,
            )

            delta_right = next_intervals - shifted_intervals

            term1 = (x_unsqueezed - left_intervals) / delta_left * bases[..., :-1]
            term2 = (next_intervals - x_unsqueezed) / delta_right * bases[..., 1:]

            bases = term1 + term2

        return bases.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        # compute B-spline bases
        bases = self.b_splines(x)

        # apply spline weights
        spline_output = torch.einsum("...ib,oib->...o", bases, self.spline_weight)

        # residual connection
        base_output = F.linear(self.base_activation_fn(x), self.base_weight)
        output = spline_output + base_output

        return output


class KolmogorovArnoldNetCore(nn.Module):
    """
    Core Kolmogorov-Arnold network implementation
    Reference:
    Wang, Y., Sun, J., Bai, J., Anitescu, C., Eshaghi, M. S.,
    Zhuang, X., … Liu, Y. (2025). Kolmogorov–Arnold-Informed
    neural network: A physics-informed deep learning framework
    for solving forward and inverse problems based on
    Kolmogorov–Arnold Networks. Computer Methods in Applied
    Mechanics and Engineering, 433, 117518.

    Parameters
    ----------
    in_features : int
        Input dimension
    out_features : int
        Output dimension
    layer_size : int, optional
        Hidden layer size, by default 5
    nr_layers : int, optional
        Number of hidden layers, by default 2
    grid_size : int, optional
        Grid size for B-splines, by default 10
    spline_order : int, optional
        Order of B-splines, by default 3
    base_activation_fn : Activation, optional
        Base activation function, by default Activation.TANH
    grid_range : Tuple[float, float], optional
        Grid range for B-splines, by default (-1, 1)
    free_knot : bool, optional
        Apply learnable biases to grid for non-uniformity, by default False
    outer_layer_fn : nn.Module, optional
        Outer layer applied to spline output, by default None
    outer_layer : bool, optional
        Apply outer layer to spline output, by default True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_size: int = 5,
        nr_layers: int = 2,
        grid_size: int = 10,
        spline_order: int = 3,
        base_activation_fn: Activation = Activation.TANH,
        grid_range: Tuple[float, float] = (-1, 1),
        free_knot: bool = False,
        outer_layer_fn: Optional[nn.Module] = None,
        outer_layer: bool = True,
    ):
        super().__init__()
        self.outer_layer = outer_layer

        # get activation function
        if isinstance(base_activation_fn, str):
            activation_fn = get_activation_fn(Activation[base_activation_fn.upper()])
        elif isinstance(base_activation_fn, Activation):
            activation_fn = get_activation_fn(base_activation_fn)
        else:
            # already an nn.Module
            activation_fn = base_activation_fn

        # get outer layer function
        if outer_layer_fn is None or isinstance(outer_layer_fn, str):
            self.outer_layer_fn = nn.Tanh()
        else:
            self.outer_layer_fn = outer_layer_fn

        # build KAN layers
        self.layer_sizes = [in_features] + [layer_size] * nr_layers + [out_features]
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(
                KANLayer(
                    in_features=self.layer_sizes[i],
                    out_features=self.layer_sizes[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation_fn=activation_fn,
                    grid_range=grid_range,
                    free_knot=free_knot,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)

            # apply outer layer
            if self.outer_layer:
                x = self.outer_layer_fn(x)

        # final layer without tanh; output can be arbitrary range
        x = self.layers[-1](x)
        return x


class KolmogorovArnoldNetArch(Arch):
    """
    Kolmogorov-Arnold Network architecture
    Reference:s
    Wang, Y., Sun, J., Bai, J., Anitescu, C., Eshaghi, M. S.,
    Zhuang, X., … Liu, Y. (2025). Kolmogorov–Arnold-Informed
    neural network: A physics-informed deep learning framework
    for solving forward and inverse problems based on
    Kolmogorov–Arnold Networks. Computer Methods in Applied
    Mechanics and Engineering, 433, 117518.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int, optional
        Layer size for every hidden layer, by default 5
    nr_layers : int, optional
        Number of hidden layers, by default 2
    grid_size : int, optional
        Number of grid points for B-spline interpolation, by default 10
    spline_order : int, optional
        Order of B-splines (3 for cubic splines), by default 3
    base_activation_fn : Activation, optional
        Base activation function for residual connections, by default Activation.TANH
    grid_range : Tuple[float, float], optional
        Range for grid initialization, by default (-1, 1)
    domain_bounds : Tuple[Tuple[float, float], ...], optional
        Domain bounds for each dimension as ((x_min, x_max), (y_min, y_max), ...),
        assumes [-1, 1] if none, by default None.
    free_knot: bool, optional
        Apply learnable biases to grid for non-uniformity, by default False
    outer_layer_fn : Activation, optional
        Outer layer applied to spline output (set as None to deactivate), by default Activation.TANH
    outer_layer : bool, optional
        Apply outer layer to spline output, by default True
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 5,
        nr_layers: int = 2,
        grid_size: int = 10,
        spline_order: int = 3,
        base_activation_fn: Activation = Activation.TANH,
        grid_range: Tuple[float, float] = (-1, 1),
        domain_bounds: Optional[List[Any]] = None,
        free_knot: bool = False,
        outer_layer_fn: Activation = Activation.TANH,
        outer_layer: bool = True,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
        )

        # Store hyperparameters
        self.layer_size = layer_size
        self.nr_layers = nr_layers
        self.grid_size = grid_size
        self.spline_order = spline_order

        # handle base activation function
        if isinstance(base_activation_fn, str):
            self.base_activation_fn = Activation[base_activation_fn.upper()]
        else:
            self.base_activation_fn = base_activation_fn

        # handle outer layer
        if isinstance(outer_layer_fn, str):
            outer_layer_fn = Activation[outer_layer_fn.upper()]

        # get outer layer
        if isinstance(outer_layer_fn, str):
            outer_layer_fn = get_activation_fn(Activation[outer_layer_fn.upper()])
        elif isinstance(outer_layer_fn, Activation):
            outer_layer_fn = get_activation_fn(outer_layer_fn)

        self.outer_layer_fn = outer_layer_fn

        self.grid_range = grid_range

        # calculate input and output dimensions
        in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        # initialize buffers for input normalization
        # set domain bounds; default to [-1,1]
        if domain_bounds is None:
            domain_bounds = torch.tensor([[-1.0, 1.0]] * in_features)
        else:
            domain_bounds = torch.tensor(domain_bounds)

        self.register_buffer("input_min", domain_bounds[:, 0])
        self.register_buffer("input_max", domain_bounds[:, 1])

        # create core network
        self._impl = KolmogorovArnoldNetCore(
            in_features=in_features,
            out_features=out_features,
            layer_size=layer_size,
            nr_layers=nr_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation_fn=self.base_activation_fn,
            grid_range=grid_range,
            free_knot=free_knot,
            outer_layer_fn=self.outer_layer_fn,
            outer_layer=outer_layer,
        )

    def _tensor_forward(self, x: Tensor) -> Tensor:
        # normalize input to [-1, 1]
        denom = self.input_max - self.input_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        x_norm = ((x - self.input_min) / denom) * 2 - 1

        # apply KAN core
        output = self._impl(x_norm)
        return output

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)


@dataclass
class KolmogorovArnoldNetConf(ModelConf):
    arch_type: str = "kan"
    layer_size: int = 5
    nr_layers: int = 2
    grid_size: int = 10
    spline_order: int = 3
    base_activation_fn: str = "tanh"
    grid_range: Tuple = (-1, 1)
    domain_bounds: Optional[List[Any]] = None
    free_knot: bool = False
    outer_layer_fn: str = "tanh"
    outer_layer: bool = True


class SeparableNetArch(Arch):
    """
    Separable neural network based on CP (CANDECOMP/PARAFAC) tensor
    decomposition
    Reference:
    Cho, J., Nam, S., Yang, H., Yun, S.-B., Hong, Y., & Park, E.
    (2023). Separable physics-informed neural networks. Proceedings
    of the 37th International Conference on Neural Information
    Processing Systems. Presented at the New Orleans, LA, USA. Red
    Hook, NY, USA: Curran Associates Inc.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list.
    output_keys : List[Key]
        Output key list.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int, optional
        Layer size for every hidden layer of each subnetwork, by default 512
    nr_layers : int, optional
        Number of hidden layers of each subnetwork, by default 6
    rank : int, optional
        Number of output features of each subnetwork
    subnet_arch_type : str, optional
        Name of architecture class to use for subnetworks, by default 'fully_connected'
    subnet_kwargs : dict, optional
        Additional keyword arguments to pass to subnet architecture, by default {}
    activation_fn : Activation, optional
        Activation function used by network, by default :obj:`Activation.SILU`
    skip_connections : bool, optional
        Apply skip connections in subnetworks, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default True
    adaptive_activations : bool, optional
        Use adaptive activation functions, by default False
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        rank: int = 64,
        subnet_arch_type: str = "fully_connected",
        subnet_kwargs: dict = {},
        activation_fn: Activation = Activation.SILU,
        skip_connections: bool = False,
        weight_norm: bool = True,
        adaptive_activations: bool = False,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
        )

        # ensure input keys are scalar
        for key in input_keys:
            if key.size != 1:
                raise ValueError(
                    "SeparableNetArch requires scalar input keys. "
                    f"Key '{key.name}' has size {key.size}"
                )

        # calculate input and output dimension
        self.rank = rank
        self.input_dim = len(input_keys)
        self.output_dim = sum(self.output_key_dict.values())

        # output dim should divide rank
        if self.output_dim > 1 and rank % self.output_dim != 0:
            raise ValueError(
                f"For vector outputs (dim={self.output_dim}), rank ({rank}) "
                f"should be divisible by output dimension"
            )

        self.rank_per_output = rank // self.output_dim if self.output_dim > 1 else rank

        # get subnet arch class
        pn_models = PhysicsNeMoModels()
        if subnet_arch_type not in pn_models:
            raise ValueError(f"Architecture '{subnet_arch_type}' not found in registry")
        subnet_arch = pn_models[subnet_arch_type]

        # make subnetwork for each input dimension
        self.subnetworks = nn.ModuleList()
        for i, input_key in enumerate(input_keys):
            sub_output_keys = [
                Key(f"{input_key.name}_feature_{j}", size=1) for j in range(rank)
            ]
            subnet = subnet_arch(
                input_keys=[input_key],
                output_keys=sub_output_keys,
                detach_keys=[],
                layer_size=layer_size,
                nr_layers=nr_layers,
                activation_fn=activation_fn,
                skip_connections=skip_connections,
                weight_norm=weight_norm,
                adaptive_activations=adaptive_activations,
                **subnet_kwargs,
            )
            self.subnetworks.append(subnet)

        if self.output_dim > 1:
            self.linear_head = nn.Linear(self.rank, self.output_dim, bias=False)

    def _tensor_forward(self, x: Tensor) -> Tensor:
        # sum over products of features from each dimension
        output = self.subnetworks[0]._tensor_forward(x[..., 0:1])
        for i in range(1, self.input_dim):
            output = output * self.subnetworks[i]._tensor_forward(x[..., i : i + 1])

        # sum across rank dimension for scalar output
        # or use learned weights for multi-output
        if self.output_dim == 1:
            return output.sum(dim=-1, keepdim=True)
        else:
            # reshape rank into output_dim groups and sum within each
            return self.linear_head(output)

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)


@dataclass
class SeparableNetConf(ModelConf):
    arch_type: str = "spinn"
    layer_size: int = 512
    nr_layers: int = 6
    rank: int = 64
    subnet_arch_type: str = "fully_connected"
    subnet_kwargs: dict = field(default_factory=dict)
    skip_connections: bool = False
    activation_fn: str = "silu"
    adaptive_activations: bool = False
    weight_norm: bool = True


def register_custom_arch_configs():
    cs = ConfigStore.instance()

    register_arch(FiniteBasisNetArch, "fbpinn")
    cs.store(
        group="arch",
        name="fbpinn",
        node={"fbpinn": FiniteBasisNetConf()},
    )

    register_arch(KolmogorovArnoldNetArch, "kan")
    cs.store(
        group="arch",
        name="kan",
        node={"kan": KolmogorovArnoldNetConf()},
    )

    register_arch(SeparableNetArch, "spinn")
    cs.store(
        group="arch",
        name="spinn",
        node={"spinn": SeparableNetConf()},
    )
