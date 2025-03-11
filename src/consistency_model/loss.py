from collections.abc import Sequence
from typing import Tuple, Union

import torch.nn.functional as F
import torch
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        *,
        net_type: Union[str, Tuple[str, ...]] = "vgg",
        l1_weight: float = 1.0,
        **kwargs):
        """ Loss combining LPIPS and L1 norm.

        Args:
            net_type: Network type used for LPIPS.
            l1_weight: Weight factor of the L1 norm in the loss.
        """

        super().__init__()

        available_net_types = ("vgg", "alex", "squeeze")

        def _append_net_type(net_type: str):
            if net_type in available_net_types:
                self.lpips_losses.append(
                    LearnedPerceptualImagePatchSimilarity(net_type)
                )
            else:
                raise TypeError(
                    f"'net_type' should be on of {available_net_types}, got {net_type}"
                )

        self.lpips_losses = nn.ModuleList()

        if isinstance(net_type, str):
            _append_net_type(net_type)

        elif isinstance(net_type, Sequence):
            for _net_type in sorted(net_type):
                _append_net_type(_net_type)

        self.lpips_losses.requires_grad_(False)

        self.l1_weight = l1_weight


    def forward(self, input: torch.tensor, target: torch.tensor):
        """ 
        Args:
            input: Network prediction.
            target: Ground truth.
        
        """
        input =  input.clamp(-1.0, 1.0)
        target =  target.clamp(-1.0, 1.0)

        upscaled_input = F.interpolate(input, (224, 224), mode="bilinear")
        upscaled_target = F.interpolate(target, (224, 224), mode="bilinear")

        if upscaled_input.shape[1] == 1 and upscaled_target.shape[1] == 1:
            upscaled_input = torch.concat([upscaled_input,upscaled_input,upscaled_input], dim=1)
            upscaled_target = torch.concat([upscaled_target,upscaled_target,upscaled_target], dim=1)

        lpips_loss = sum(
            _lpips_loss(upscaled_input, upscaled_target)
            for _lpips_loss in self.lpips_losses
        )

        return lpips_loss + self.l1_weight * F.l1_loss(input, target)


LPIPSLoss = PerceptualLoss
