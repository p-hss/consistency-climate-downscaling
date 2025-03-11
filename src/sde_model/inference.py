import torch
import numpy as np
import xarray as xr
from pathlib import Path
#from tqdm.notebook import tqdm
from tqdm import tqdm
from typing import List, Optional, Type, Union, Tuple
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt

from src.utils.transforms import apply_inverse_transforms
from src.sde_model.model import SDEModel
from src.configuration import Config
import src.utils.xarray_utils as xu

class Inference:
    def __init__(self,
                 config: Config):
        """Evaluates the trained score model.
        
        Args:
            config: Model configuration.
        """
        
        self.config = config
        self.batch_size = config.batch_size
        self.sample_dimension = config.sample_dimension 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.results = None
        self.checkpoint = None
        self.training_target = None
        self.test_target = None
        self.test_input = None
        
        
    def load_model(self,
                   checkpoint_fname: str = 'best'):
        """ Loads the model from a checkpoint.

        Args:
            checkpoint_fname: Path to the .ckpt file. Default 'best' loads .../best_model.ckpt
        """

        if checkpoint_fname == 'best':
            self.checkpoint_path = f'{self.config.checkpoint_path}/best_model.ckpt'
        else:
            self.checkpoint_path = f'{self.config.checkpoint_path}/{checkpoint_fname}'
        assert Path(self.checkpoint_path).exists(), f"Path {self.checkpoint_path} does not exist."

        self.checkpoint = torch.load(self.checkpoint_path)

        model_hyperparameters = ['channels', 'down_block_types', 'up_block_types', 'diffusion_model',
                                 'sigma', 'sigma_max', 'sigma_min', 'epsilon']

        config_checkpoint = {}
        for key in self.checkpoint['hyper_parameters'].keys(): 

            if key in model_hyperparameters:
                setattr(self.config, key, self.checkpoint['hyper_parameters'][key])

            config_checkpoint[key] = self.checkpoint['hyper_parameters'][key]
            
        self.model = SDEModel.load_from_checkpoint(self.checkpoint_path,
                                                   config=self.config)

        self.model.config_checkpoint = config_checkpoint

        self.model.to(self.device)
        self.model.eval()


    def load_data(self,
                      training_target: xr.DataArray,
                      test_target: xr.DataArray,
                      test_input: xr.DataArray ):
        """ Loads datasets.
        
        Args:
            training_target: The ground truth training dataset.
            test_target: The target test set for comparisons.
            test_input: The input to be downscaled, e.g. the ESM test set.
        """

        self.training_target = training_target
        self.test_target = test_target
        self.test_input = test_input
        
        
    def run(self,
            sampler_type="sde",
            num_steps=500,
            num_batches=1,
            convert_to_xarray=True,
            inverse_transform=True,
            show_progress=False, 
            init_x: Optional[torch.tensor]=None) -> np.ndarray:
        """Executes the inference sampling.
        
        Args:
            sampler_type: "sde" 
            num_steps: Number of integration steps
            convert_to_xarray: Converts torch tensor to xarray's format
            inverse_transform: Either "sde" or "ode"
            show progress: Displays a progress bar
            init_x: Initial condition for the SDE, randomly generated if None is provided 

        Returns:
            The downscaled fields in physical units.
        """
        
        all_samples = []
        assert sampler_type in ['sde', 'ode', 'ode_gpu', 'pc'], "sampler type {sampler_type} can be sde, ode or pc."

        num_batches = range(num_batches)

        if show_progress:
            num_batches = tqdm(num_batches)

        for i in num_batches:

            if sampler_type == 'sde':
                samples = self.model.euler_maruyama_sampler(batch_size=self.batch_size,
                                                            sample_dimension=self.sample_dimension, 
                                                            init_x=init_x,
                                                            num_steps=num_steps)

            all_samples.append(samples)

        all_samples = torch.cat(all_samples).cpu().numpy()

        if all_samples.shape[-2] == 64:
            all_samples = all_samples[:,:,2:62] # remove padding

        if convert_to_xarray:
            all_samples = self.convert_to_xarray(all_samples)

        if inverse_transform:
            all_samples = apply_inverse_transforms(all_samples,
                                                   self.training_target,
                                                   self.config)
        self.results = all_samples 
        return self.results


    def run_bridge(self,
                   esm_dataloader,
                   reverse_num_steps: int=500,
                   forward_num_steps: int=500,
                   num_batches=1,
                   stop_step: int=np.inf,
                   convert_to_xarray: bool=True,
                   inverse_transform: bool=True) -> np.ndarray:
        """Executes the inference sampling by noising an upsampled ESM field and
        then denoising it with a reverse SDE.
        
        Args:
            forward_num_steps: number of steps for forward integration
            reverse_num_steps: number of steps for reverse integration
            
        Returns:
            The downscaled fields in physical units, the noised and raw ESM fields.
        """
 
        all_esm = []
        all_samples = []
        all_conditions = []

        for b in range(num_batches):
            for x in tqdm(esm_dataloader):

                init_x = x.to(self.device)
                if len(init_x.shape) == 3:
                    init_x = init_x.unsqueeze(1)
                conditionings = self.model.conditional_euler_maruyama_sampler(batch_size=init_x.shape[0],
                                                                        sample_dimension=(init_x.shape[-2],init_x.shape[-1]),
                                                                        init_x=init_x,
                                                                        num_steps=forward_num_steps,
                                                                        stop_step=stop_step,
                                                                        forward=True)

                samples = self.model.conditional_euler_maruyama_sampler(batch_size=init_x.shape[0],
                                                                        sample_dimension=(init_x.shape[-2],init_x.shape[-1]),
                                                                        init_x=conditionings,
                                                                        num_steps=reverse_num_steps,
                                                                        stop_step=stop_step,
                                                                        forward=False)
                all_esm.append(init_x)
                all_samples.append(samples)
                all_conditions.append(conditionings)

        all_esm = torch.cat(all_esm).cpu().numpy()
        all_samples = torch.cat(all_samples).cpu().numpy()
        all_conditions = torch.cat(all_conditions).cpu().numpy()

        if all_esm.shape[-2] == 64:
            all_esm = all_esm[:,:,2:62,:] # remove padding
            all_samples= all_samples[:,:,2:62,:] # remove padding
            all_conditions = all_conditions[:,:,2:62,:] # remove padding

        if convert_to_xarray:
            all_esm = self.convert_to_xarray(all_esm)
            all_samples = self.convert_to_xarray(all_samples)
            all_conditions = self.convert_to_xarray(all_conditions)

        else:
            all_esm = all_esm
            all_samples = all_samples
            all_conditions = all_conditions

        if inverse_transform:
            all_esm = apply_inverse_transforms(all_esm,
                                               self.training_target,
                                               self.config)
            all_samples = apply_inverse_transforms(all_samples,
                                                   self.training_target,
                                                   self.config)
            all_conditions = apply_inverse_transforms(all_conditions,
                                                   self.training_target,
                                                   self.config)
    
        return {'generated': all_samples, 'conditions': all_conditions, 'esm': all_esm}
    
    
    def convert_to_xarray(self,
                          samples: np.ndarray) -> xr.DataArray:
        """ Transforms the samples tensors to xarray format. """

        if len(samples.shape) == 4:
            samples = samples[:,0]

        results = xr.DataArray(
            data=samples,
            dims=["time", "latitude", "longitude"],
            coords=dict(
                time=self.test_input.time[0:len(samples)],
                latitude=self.test_input.latitude,
                longitude=self.test_input.longitude
            ),
            attrs=dict(
                description=self.config.predict_variable
                )
        )
        
        return results


