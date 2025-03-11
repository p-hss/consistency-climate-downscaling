from typing import List, Optional, Type, Union, Tuple
import numpy as np
import torch 
import pickle
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr

from src.inference import Inference
from src.data import GeoDataset
import src.utils.xarray_utils as xu
from src.utils.transforms import apply_transforms
from src.configuration import Config
from src.utils.spectra import mean_rapsd
from src.utils.utils import time_to_steps


class Experiment():
    
    def __init__(self,
                 config: Config,
                 num_sde_steps: Optional[int]=500) -> None:
        """ Collects methods for data and model loading, inference and plotting. """
        
        self.config = config
        self.samples = {}
        self.num_sde_steps = num_sde_steps
        self.inference = Inference(config)


    def prepare_data(self, lazy: Optional[bool]=True) -> None:
        """ Load datasets from file and preprocesses them."""

        self.config.lazy = lazy
        
        self.era5_train = GeoDataset("train", "ERA5", self.config).target.astype(np.float32)
        self.era5_test = GeoDataset("test", "ERA5", self.config).target.astype(np.float32)

        self.esm_train = GeoDataset("train", "ESM", self.config).climate_model.astype(np.float32)
        self.esm_test = GeoDataset("test", "ESM", self.config).climate_model.astype(np.float32)

        self.inference.load_data(training_target=self.era5_train,
                       test_target=self.era5_test,
                       test_input=self.esm_test)


    def load_model(self, checkpoint_fname: str) -> None:
        """ Load model checkpoint from file. """

        self.inference.load_model(checkpoint_fname=checkpoint_fname)
        
        
    def sample_unconditional(self, show_progress: Optional[bool]=True) -> None:
        """ Generates unconditional samples. """

        self.inference.sample_dimension = (len(self.era5_test.latitude.values), len(self.era5_test.longitude.values))
    
        self.samples['unconditional'] = self.inference.run(sampler_type='sde',
                                                           num_steps=self.num_sde_steps,
                                                           convert_to_xarray=True,
                                                           inverse_transform=True,
                                                           show_progress=show_progress)

    def sample_bridge(self,
                      noise_times: list,
                      esm_initial_condition: torch.tensor) -> None:
        """ Uses the SDE brige for sampling.

        Args:
            noise_times: list of times to terminate the forward SDE.
            esm_initial_condition: torch tensor containing ESM fields
            interpolated to the diffusion model resolution of the shape [batch, channel, height, width]
        """

        for noise_time in noise_times:

            stop_step = time_to_steps(noise_time, self.num_sde_steps)

            results = self.inference.run_bridge(esm_dataloader=esm_initial_condition,
                                                reverse_num_steps=self.num_sde_steps,
                                                forward_num_steps=self.num_sde_steps,
                                                stop_step=stop_step,
                                                convert_to_xarray=True,
                                                inverse_transform=True)

            self.samples[noise_time] = results


    def plot_sample(self) -> None:
        """ Plots a generated and target sample. """

        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.title("Sample (uncond.)")
        plt.imshow(self.samples['unconditional'][0], origin='lower', vmax=0.0002)
        
        plt.subplot(1,2,2)
        plt.title("Target")
        plt.imshow(self.era5_test[0], origin='lower', vmax=0.0002)

        plt.show()

    
    def save(self, fname: str) -> None:
        """ Saves dictionary with results to disk. """

        for key in self.samples.keys():
            self.samples[key].load()

        with open(fname, 'wb') as f:
            pickle.dump(self.samples, f)


    def save_netcdf(self, fname: str, key=None) -> None:
        """ Saves a single dictionary entry (xarray dataset) to disk as netcdf. """

        if key is None:
            xu.write_dataset(self.samples['unconditional'].rename("precipitation"), fname)
        else:
            xu.write_dataset(self.samples[key]['generated'].rename("precipitation"), fname)
            
            
    def load(self, fname: str) -> None:
        """ Loads saved dictionary from a given file. """

        with open(fname, 'rb') as handle:
            self.samples = pickle.load(handle)


    def compute_spectra(self) -> None:
        """ Computes radially averaged power spectral densities of:
            - the ESM data
            - the ERA5 target data.
        """
        
        num_latitudes = len(self.era5_test.latitude)
        offset = num_latitudes//2

        esm = apply_transforms(self.esm_test[:,:,offset:num_latitudes+offset],
                               data_ref=self.esm_train,
                               config=self.config).load()
        self.esm_psd = mean_rapsd(esm, normalize=True)

        era5 = apply_transforms(self.era5_test[:,:,offset:num_latitudes+offset],
                                data_ref=self.era5_train,
                                config=self.config).load()
        self.era5_psd = mean_rapsd(era5, normalize=True)
    

    def plot_spectra(self,
                     freq_min: Optional[float]=None,
                     psd_val: Optional[float]=None,
                     fname: Optional[str]=None) -> None:
        """ Plots the PSDs together with the intersection frequency. """

        self.log_diff = abs(np.log(self.era5_psd[0]) - np.log(self.esm_psd[0]))[1:]
        x_min = np.where(self.log_diff==self.log_diff.min())
        if freq_min is not None:
            self.freq_min = freq_min
        else:
            self.freq_min = self.era5_psd[1][1:][x_min]

        print(f"PSD intersection at freq={self.freq_min}")

        plt.figure(figsize=(7,5))

        mpl.rcParams['axes.linewidth'] = 1.5
        plt.tick_params(width=1.5)

        plt.subplot(1,1,1)

        plt.plot(self.era5_psd[1], self.era5_psd[0], label='ERA5', c='k', lw=2)
        plt.plot(self.esm_psd[1], self.esm_psd[0], label='POEM',  c='orange', lw=2)

        plt.axvline(x=self.freq_min, c='tab:gray', ls='--', lw=2, label=r'$k^* = $'+f'{self.freq_min[0]:2.4f}')
        if psd_val is not None:
            plt.axhline(y=psd_val, c='tab:gray', ls='-', lw=2, label=r'$\mathrm{PSD}(k^*) = $'+f'{psd_val:2.1e}')

        plt.yscale("log", base=2)
        plt.xscale("log", base=2)
        plt.xlabel(r'Wavenumber')
        plt.ylabel('Power spectral density')
        plt.ylim(2**(-30), 2**(-8))
        plt.legend(frameon=False)

        if fname is not None:
            plt.savefig(fname, format='pdf', bbox_inches='tight')
            print(fname)
        plt.show()





