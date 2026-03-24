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


import torch
from dataclasses import dataclass
from typing import Dict

from physicsnemo.sym.loss.aggregator import Aggregator
from physicsnemo.sym.hydra.loss import LossConf
from hydra.core.config_store import ConfigStore


class BalancedResidualDecayRate(Aggregator):
    """
    Aggregate loss with adaptive weighting based on balanced residual
    decay rates (BRDR)
    Chen, W., Howard, A. A., & Stinis, P. (2025). Self-adaptive weights
    based on balanced residual decay rate for physics-informed neural
    networks and deep operator networks. Journal of Computational
    Physics, 542, 114226. doi:10.1016/j.jcp.2025.114226
    """

    def __init__(
        self, params, num_losses, weights=None, beta_c=0.999, beta_w=0.999, eps=1.0e-14
    ):
        super().__init__(params, num_losses, weights)
        self.beta_c: float = beta_c
        self.beta_w: float = beta_w
        self.eps: float = eps
        self.register_buffer(
            "residual_4th_ema", torch.zeros(self.num_losses, device=self.device)
        )
        self.register_buffer(
            "weights_ema", torch.ones(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Sums losses with self-adaptive weighting based on balanced
        residual decay rates

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """
        n = step + 1

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # compute squared residuals
        losses_stacked: torch.Tensor = torch.stack(list(losses.values()))
        residuals_squared: torch.Tensor = torch.clamp(losses_stacked, min=0.0)

        # first step logic
        if step == 0:
            self.residual_4th_ema = residuals_squared.clone().detach() ** 2
            return losses_stacked.sum()

        # compute balanced residual decay rates
        with torch.no_grad():
            residual_4th = residuals_squared**2
            self.residual_4th_ema = (
                self.beta_c * self.residual_4th_ema + (1 - self.beta_c) * residual_4th
            )

            # bias correction
            residual_4th_ema = self.residual_4th_ema / (1 - self.beta_c**n)

            # compute weights
            irdr = residuals_squared / (torch.sqrt(residual_4th_ema) + self.eps)
            weights = irdr / (irdr.mean() + self.eps)
            self.weights_ema = (
                self.beta_w * self.weights_ema + (1 - self.beta_w) * weights
            )
            self.weights_ema = torch.clamp(self.weights_ema, min=self.eps)
            self.weights_ema = (
                self.weights_ema * self.num_losses / (self.weights_ema.sum() + self.eps)
            )

        # compute total loss
        loss = (self.weights_ema.detach() * losses_stacked).sum()
        return loss


@dataclass
class BalancedResidualDecayRateConf(LossConf):
    _target_: str = "registry.loss.BalancedResidualDecayRate"
    beta_c: float = 0.999
    beta_w: float = 0.999
    eps: float = 1.0e-14


def register_custom_loss_configs():
    cs = ConfigStore.instance()

    cs.store(group="loss", name="brdr", node=BalancedResidualDecayRateConf)
