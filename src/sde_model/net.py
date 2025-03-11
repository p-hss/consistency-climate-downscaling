import torch.nn as nn

from diffusers import UNet2DModel

class ScoreUNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self,
                  marginal_prob_std,
                  resolution=64,
                  in_channels=1,
                  out_channels=1,
                  channels=(64, 64, 128, 128),
                  down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D"),
                  up_block_types=(
                    "UpBlock2D",
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                  ),
                 ):
      """Initialize a time-dependent score-based network.

      Args:
        marginal_prob_std: A function that takes time t and gives the standard
          deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
      """
      super().__init__()

      self.model = UNet2DModel(
          sample_size=resolution,
          in_channels=in_channels,
          out_channels=out_channels,
          block_out_channels=channels,
          down_block_types=down_block_types,
          up_block_types=up_block_types
          )

      self.marginal_prob_std = marginal_prob_std
  
    def forward(self, x, t): 
        """
        Args:
            x: Image tensor.
            t: Noise schedule time.
        
        Returns:
            Denoised image.
        """

        x = self.model(x, t).sample
        # Normalize output
        x = x / self.marginal_prob_std(t)[:, None, None, None]
        return x


