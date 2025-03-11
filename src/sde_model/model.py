from typing import List, Optional, Type, Union, Tuple

import numpy as np
import tqdm
from pytorch_lightning import LightningModule
import torch

from dataclasses import asdict
import matplotlib.pyplot as plt

from src.configuration import Config
from src.sde_model.ema import ExponentialMovingAverage
from src.sde_model.loss import VELoss
from src.sde_model.net import ScoreUNet

class SDEModel(LightningModule):

    def __init__(self,
                 config: Config,
                 verbose: bool = False) -> None: 
        super().__init__()
        """Includes training and inference sampling of a score based diffusion model.
        
        Args:
            config: Stores hyperparameters and file paths.
            verbose: Prints the training configuration.
        """

        self.save_hyperparameters(asdict(config), ignore=['model'])

        self.config = config
        self.config_checkpoint = None

        if verbose: 
            print('Initializing SDEModel with Network resolution ='+str(config.network_resolution),' and channels='+str(config.channels))

        self.net = ScoreUNet(marginal_prob_std=self.marginal_prob_std,
                             channels=config.channels,
                             in_channels=config.in_channels,
                             out_channels=config.out_channels,
                             resolution=config.network_resolution,
                             down_block_types=config.down_block_types,
                             up_block_types=config.up_block_types 
                             )
        
        self.loss = VELoss(marginal_prob_std=self.marginal_prob_std)

        self.current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if config.use_ema:
            self.ema = ExponentialMovingAverage(self.net.to(self.current_device).parameters(),
                                                decay=config.ema_rate)


    def configure_optimizers(self)-> torch.optim:
        return torch.optim.Adam(self.net.parameters(), lr=self.config.lr)


    def on_save_checkpoint(self, checkpoint):
        if self.config.use_ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()


    def training_step(self, x, batch_idx) -> torch.Tensor:
        """ Performs a single training step.

        Args:
            x: Input batch 

        Returns:
            Training loss
        """
        
        loss = self.loss(self.net, x)
    
        optimizer = self.optimizers()

        if self.config.warmup > 0:
            for g in optimizer.param_groups:
                  g['lr'] = self.config.lr * np.minimum(self.global_step/ self.config.warmup, 1.0)

        if self.config.use_ema: 
            self.ema.update(self.net.parameters())

        self.log("train_loss",
                  loss.detach(),
                  on_step=False,
                  on_epoch=True,
                  prog_bar=True,
                  logger=True)

        return loss


    @torch.no_grad()
    def validation_step(self, x, batch_idx)-> torch.Tensor:
        """ Performs a single validation step.

        Args:
            x: Input batch 

        Returns:
            Validation loss
        """

        loss_dict = {}

        if self.config.use_ema:

            self.ema.store(self.net.parameters())
            self.ema.copy_to(self.net.parameters())
            loss = self.loss(self.net, x)
            self.ema.restore(self.net.parameters())
            loss_dict['val_loss'] = loss.detach()

        else:

            loss = self.loss(self.net, x)
            loss_dict['val_loss'] = loss

        loss_dict['gpu-alloc'] = torch.cuda.max_memory_allocated(self.device) / 2**30
        loss_dict['gpu-reserved'] = torch.cuda.max_memory_reserved(self.device) / 2**30

        self.log_dict(loss_dict,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True)

        return loss 



    def marginal_prob_std(self, t: torch.tensor):
        """Compute the standard deviation of $p_{0t}(x(t) | x(0))$.

        Variance exploding (VE): (hyperparameters: $\sigma_{min}$ and $\sigma_{max}$)

            $\sigma^2(t) = \sigma^2_{min} (\frac{\sigma_{max}}{\sigma_{min}})^{2t}$

        Args:    
          t: A vector of time steps.
    
        Returns:
          The standard deviation.
        """   

        std = self.config.sigma_min*(self.config.sigma_max/self.config.sigma_min)**t

        return std


    def diffusion_coeff(self, t: torch.tensor):
        """Compute the diffusion coefficient of the SDE.

        Variance exploding (VE):

            $g(t) = \sigma_{min} (\frac{\sigma_{max}}{\sigma_{min}})^{t} 
            \sqrt{2 \log{\frac{\frac{\sigma_{max}}{\sigma_{min}}}}}$

        Args:
          t: A vector of time steps.
    
        Returns:
          The vector of diffusion coefficients.
        """

        coeff = self.config.sigma_min * (self.config.sigma_max/self.config.sigma_min)**t  \
                    * np.sqrt(2 * np.log(self.config.sigma_max/self.config.sigma_min))
                    
        return coeff
    
    def generate_noise(self, 
                       t: torch.tensor, 
                       batch_size: int,
                       channels: int,
                       sample_dimension: Tuple[int, int]) -> torch.Tensor:

        x_init = torch.randn(batch_size, channels, sample_dimension[0], sample_dimension[1],
                            device=self.device) *self.marginal_prob_std(t)[:, None, None, None]

        return x_init 

    @torch.no_grad()
    def euler_maruyama_sampler(
        self,
        batch_size: int,
        sample_dimension: Tuple[int, int],
        eps: Optional[float]=1e-3,
        num_steps: Optional[int]=500, 
        stop_step: Optional[float]=np.inf,
        show_progress: Optional[bool]=False,
        init_x: Optional[torch.tensor]=None 
     ) -> torch.Tensor:
        """Generate samples from score-based models with the Euler-Maruyama scheme.

        Args:
            batch_size: Number of samples in gbatch
            sample dimention: Height and width of generated image
            eps: The smallest time step for numerical stability
            num_steps: number of SDE integration steps
            stop_step: Step number that terminates the SDE integration
            show_progress: show a progress bar for sampling
            init_x: Initial condition for the SDE, randomly generated if None is provided 

        Returns:
            Generated samples
        """

        t = torch.ones(batch_size, device=self.device)

        if init_x is None: 
            init_x = self.generate_noise(t, batch_size, self.config.in_channels, sample_dimension) 

        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x

        if show_progress:
            time_steps = tqdm.notebook.tqdm(time_steps)

        for step, time_step in enumerate(time_steps):      

            batch_time_step = torch.ones(batch_size, device=self.device) * time_step
            g = self.diffusion_coeff(batch_time_step)

            mean_x = (g**2)[:, None, None, None] * self.net(x, batch_time_step)

            mean_x = x + mean_x * step_size 

            noise = torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      

            x = mean_x + noise
        
            if step > stop_step:
                break

        return mean_x 


    @torch.no_grad()
    def conditional_euler_maruyama_sampler(
        self,
        batch_size: int,
        sample_dimension: Tuple[int, int],
        init_x: Optional[torch.Tensor]=None,
        eps: Optional[float]=1e-3,
        num_steps: Optional[int]=500, 
        stop_step: Optional[int]=None,
        step_size: Optional[float]=None,
        show_progress: Optional[bool]=False,
        forward: Optional[bool]=False,
     ) -> torch.Tensor:
        """Generate samples from score-based models with the Euler-Maruyama scheme
        using a conditional starting point for the denoising.

        Args:
            batch_size: Number of samples in gbatch
            sample dimention: Height and width of generated image
            init_x: starting point for SDE integration
            eps: The smallest time step for numerical stability
            num_steps: Number of SDE integration steps
            stop_step: Step number that terminates the SDE integration
            step_size: Time step size for SDE integration
            init_x: Initial condition for the SDE, randomly generated if None is provided 
            forward: Runs SDE forward in time

        Returns:
            Generated samples
        """

        if init_x is None:
            t = torch.ones(batch_size, device=self.device)
            x = torch.randn(batch_size, 1, sample_dimension[0], sample_dimension[1], device=self.device) \
            * self.marginal_prob_std(t)[:, None, None, None] 
        else:
            mean_x = x = init_x

        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        if stop_step is not None:
            time_steps = time_steps[-stop_step:]

        if forward:
            time_steps = torch.linspace(1., eps, num_steps, device=self.device).flip(dims=(0,))
            if stop_step is not None:
                time_steps = time_steps[:stop_step]

        if step_size is None:
            step_size = abs(time_steps[0] - time_steps[1])

        if show_progress:
            time_steps = tqdm.notebook.tqdm(time_steps)
      
        for i, time_step in enumerate(time_steps):      

            batch_time_step = torch.ones(batch_size, device=self.device) * time_step
            g = self.diffusion_coeff(batch_time_step)

            if forward:
                mean_x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(mean_x)      
            else:
                mean_x = (g**2)[:, None, None, None] * self.net(x, batch_time_step)

                mean_x = x + mean_x * step_size 

                noise = torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      

                x = mean_x + noise

        # Do not include any noise in the last sampling step.
        return mean_x

