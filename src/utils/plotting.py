import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import functools
import skimage
import tqdm
from scipy.stats import pearsonr
from skimage.measure import block_reduce
import matplotlib as mpl
import matplotlib.ticker as ticker
from dask.diagnostics import ProgressBar

import src.utils.xarray_utils as xu
from src.configuration import Config
from src.utils.spectra import mean_rapsd


class PlotResults():
    """Contains plotting functions for reproducing published figures."""

    def __init__(self, out_path):

        self.config = Config(out_path=out_path)
        self.out_path = out_path + 'figures/'

        self.plot_config = {
                            'era5': {'label': 'ERA5', 'color': 'black', 'lw': 2.5, 'alpha': 0.8},
                            'esm_lr': {'label': 'POEM (raw)', 'color': 'grey', 'lw': 2.5, 'alpha': 0.8},
                            'esm': {'label': 'POEM', 'color': 'orange', 'lw': 2.5, 'alpha': 0.8},
                            'sde': {'label': 'SDE', 'color': 'tab:cyan', 'lw': 2.5, 'alpha': 0.8},
                            'sde_lr': {'label': 'SDE (upscaled)', 'color': 'tab:cyan', 'lw': 2.5, 'alpha': 0.8},
                            'cm': {'label': 'CM', 'color': 'magenta', 'lw': 2.5, 'alpha': 0.8},
                            'cm_lr': {'label': 'CM (upscaled)', 'color': 'magenta', 'lw': 2.5, 'alpha': 0.8},
                            }

        self.test_split = slice('2004', '2018')
        self.train_split = slice('1950', '1990')

        self.projection = 'robin'
        self.lon_0 = 0


    def load_data(self):

        sec_to_day = 24*3600

        # downloaded
        self.config.target_filename = 'era5_4x_downscaled_v3.nc'
        self.era5_train = xr.open_dataset(self.config.data_path + self.config.target_filename, chunks={'time': 1}).precipitation.sel(time=self.train_split)*(24*3600)
        self.era5 = xr.open_dataset(self.config.data_path + self.config.target_filename).precipitation.sel(time=self.test_split)*sec_to_day

        # downloaded
        self.config.esm_filename = 'poem_historical_3deg_4xdown_bilinear_lowpass_dqm.nc'
        self.esm_hr_train = xr.open_dataset(self.config.data_path + self.config.esm_filename, chunks={'time': 1}).precipitation.sel(time=self.train_split)*(24*3600)
        esm_hr = xr.open_dataset(self.config.data_path + self.config.esm_filename).precipitation.sel(time=self.test_split)*sec_to_day
        self.esm_hr = xu.shift_longitudes(esm_hr)

        # lrz
        self.config.esm_filename = 'poem_historical_3deg_4xdown_bilinear_lowpass_v3.nc'
        esm_hr_no_qm = xr.open_dataset(self.config.data_path +  self.config.esm_filename).precipitation.sel(time=self.test_split)*sec_to_day
        self.esm_hr_no_qm = xu.shift_longitudes(esm_hr_no_qm)

        # downloaded
        esm_lr_filename = 'poem_historical_3deg.nc'
        esm_lr = xr.open_dataset(self.config.data_path + esm_lr_filename, chunks={'time': 1}).precipitation.sel(time=self.test_split)*sec_to_day
        self.esm_lr = xu.shift_longitudes(esm_lr)

        # lrz
        sde_filename = 'sde_historical_poem_4x_dqm_2004_2018.nc'
        sde = xr.open_dataset(self.out_path + sde_filename, chunks={'time': 1}).precipitation.sel(time=self.test_split)*sec_to_day
        self.sde = xu.shift_longitudes(sde)

        # downloaded
        cm_filename = 'cm_historical_scale_468.nc'
        cm = xr.open_dataset(self.out_path + cm_filename, chunks={'time': 1}).precipitation*sec_to_day
        self.cm = xu.shift_longitudes(cm)

        cm_filename = 'cm_historical_scale_468_ensemble.nc'
        cm_ens = xr.open_dataset(self.out_path + cm_filename).precipitation*sec_to_day
        self.cm_ens = xu.shift_longitudes(cm_ens)

        # downloaded
        fname_trend_all = 'cm_trend_*'
        self.cm_trend = xr.open_mfdataset(self.out_path + fname_trend_all).sel(time=slice("2020", "2100"))*sec_to_day

        # pik
        fname_dqm_full = 'poem_full_3deg_4xdown_bilinear_lowpass_dqm.nc'
        self.dqm_ssp = xr.open_dataset(self.out_path + fname_dqm_full, chunks={'time': 1}).sel(time=slice("2020", "2100"))*sec_to_day


    def single_field(self, input, lats, lons, ax, cfg, vmin=0.1, vmax=20, plot_mask=True, cmap='YlGnBu'):

        Lon, Lat = np.meshgrid(lons, lats)
        m = Basemap(llcrnrlon=lons[0], llcrnrlat=lats[0],
                    urcrnrlon=lons[-1], urcrnrlat=lats[-1],
                    projection=self.projection, lon_0=self.lon_0, 
                    resolution='l', ax=ax)
        m.drawcoastlines(color='grey', linewidth=0.5)
        x, y = m(Lon, Lat)
    
        ax.set_title(cfg['label'])
        cs = ax.pcolormesh(x, y, input, vmin=vmin, vmax=vmax,
                                alpha=cfg['alpha'], cmap=cmap,
                                linewidth=0, shading='auto')
        if plot_mask:
            mask = np.ma.masked_where(input > 1, input)
            ax.pcolormesh(x,y, mask, vmin=-1, vmax=-1, alpha=1.0, cmap='Greys',shading='auto')
        return cs

    
    def pool(self, x, stride):
        return block_reduce(x, (stride,stride), np.mean)


    def correlation(self, a, b):
        return pearsonr(a.flatten(), b.flatten()).statistic


    def plot_all_single_fields(self):

        idx = 51

        fname = self.out_path + 'global_fields_old_cmap_large.pdf'

        fig = plt.figure(figsize=(9,11.8), constrained_layout=True)
        mpl.rcParams['axes.linewidth'] = .5

        subfig = fig.subfigures(nrows=1, ncols=1)
        axes = subfig.subplots(nrows=4, ncols=2, sharey=True)

        # era5 hr
        cfg = self.plot_config['era5']
        cfg['label'] = 'ERA5 ('+r'$4 \times \mathrm{r}_{\mathrm{POEM}}$'+')'
        ax = axes[0,0]
        ax.annotate("A", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        data = self.era5[idx]
        lats = data.latitude
        lons = data.longitude
        self.single_field(data, lats, lons, ax, cfg)

        # era5 lr
        cfg = self.plot_config['era5']
        cfg['label'] = 'ERA5 ('+r'$\mathrm{r}_{\mathrm{POEM}} = 3^\circ \times 3.75^\circ$'+')'
        ax = axes[0,1]
        ax.annotate("B", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        data = self.era5[idx]
        data = self.pool(data.values, stride=4)
        lats = self.esm_lr.latitude
        lons = self.esm_lr.longitude
        self.single_field(data, lats, lons, ax, cfg)

        # esm hr
        cfg = self.plot_config['esm']
        cfg['label'] = 'Interpolation ('+r'$4 \times \mathrm{r}_{\mathrm{POEM}}$'+')'
        ax = axes[1,0]
        ax.annotate("C", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        data = self.esm_hr[idx]
        lats = self.esm_hr.latitude
        lons = self.esm_hr.longitude
        self.single_field(data, lats, lons, ax, cfg)

        # esm lr
        cfg = self.plot_config['esm']
        cfg['label'] = 'POEM ('+r'$\mathrm{r}_{\mathrm{POEM}}$'+')' 
        ax = axes[1,1]
        ax.annotate("D", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        data = self.esm_hr[idx]
        data = self.pool(data.values, stride=4)
        lats = self.esm_lr.latitude
        lons = self.esm_lr.longitude
        self.single_field(data, lats, lons, ax, cfg)

        # sde hr
        cfg = self.plot_config['sde']
        cfg['label'] = 'SDE ('+r'$4 \times \mathrm{r}_{\mathrm{POEM}}$'+')'
        ax = axes[2,0]
        ax.annotate("E", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        data = self.sde[idx]
        lats = data.latitude
        lons = data.longitude
        self.single_field(data, lats, lons, ax, cfg)

        # sde lr
        cfg = self.plot_config['sde_lr']
        ax = axes[2,1]
        ax.annotate("F", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 

        data = self.sde[idx]
        sde_data = self.pool(data.values, stride=4)
        data = self.esm_hr[idx]
        esm_data = self.pool(data.values, stride=4)
        corr = self.correlation(sde_data, esm_data)
        cfg['label'] = 'SDE ('+r'$\mathrm{r}_{\mathrm{POEM}}$'+')' + f', corr'+r'$_{\mathrm{POEM}}$'+f'= {corr:2.2f}'

        lats = self.esm_lr.latitude
        lons = self.esm_lr.longitude
        self.single_field(sde_data, lats, lons, ax, cfg)

        # cm hr
        cfg = self.plot_config['cm']
        cfg['label'] = 'CM ('+r'$4 \times \mathrm{r}_{\mathrm{POEM}}$'+')'

        ax = axes[3,0]
        ax.annotate("G", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        data = self.cm[idx]
        lats = data.latitude
        lons = data.longitude
        cs = self.single_field(data, lats, lons, ax, cfg)

        # cm lr
        ax = axes[3,1]
        ax.annotate("H", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        cfg = self.plot_config['cm_lr']

        data = self.cm[idx]
        cm_data = self.pool(data.values, stride=4)
        data = self.esm_hr[idx]
        esm_data = self.pool(data.values, stride=4)
        corr = self.correlation(cm_data, esm_data)
        cfg['label'] = 'CM ('+r'$\mathrm{r}_{\mathrm{POEM}}$'+')' + f', corr'+r'$_{\mathrm{POEM}}$'+f'= {corr:2.2f}'

        lats = self.esm_lr.latitude
        lons = self.esm_lr.longitude
        self.single_field(cm_data, lats, lons, ax, cfg)

        cbar = subfig.colorbar(cs, ax=axes, location='bottom', shrink=0.3, extend='max', label="Precipitation [mm/day]")
        cbar.set_ticks(ticks=[1,5,10,15,20])

        plt.savefig(fname, format='png',  bbox_inches='tight', dpi=300)
        print(fname)


    def compute_power_spectral_density_for_noise_time(self, time):

        fname = self.out_path+f'cm_historical_scale_{int(time*1000)}.nc'
        cm_scale = xr.open_dataset(fname, chunks={"time": 10}).precipitation
        cm_scale = xu.shift_longitudes(cm_scale).load()
    
        num_latitudes = len(cm_scale.latitude)
        offset = num_latitudes//2
    
        psd = mean_rapsd(cm_scale[:,:,offset:num_latitudes+offset], normalize=True)

        return psd


    def compute_power_spectral_density_over_scales(self):

        self.time_list = [0.002, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.25, 0.35, 0.468,
                          0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.5, 5, 7.5, 10, 15, 25, 50, 80]

        psd_scale = {}
        for t in self.time_list:
            psd_scale[t] = self.power_spectral_density_for_noise_time(t)

        data = (self.era5.chunk({'time': 10})/(24*3600)).load()

        num_latitudes = len(data.latitude)
        offset = num_latitudes//2
        
        era5_psd = mean_rapsd(data[:,:,offset:num_latitudes+offset], normalize=True)
        
        data = (self.esm_hr.chunk({'time': 10})/(24*3600)).load()
        esm_psd = mean_rapsd(data[:,:,offset:num_latitudes+offset], normalize=True)
        
        data = (self.sde.chunk({'time': 10})/(24*3600)).load()
        sde_psd = mean_rapsd(data[:,:,offset:num_latitudes+offset], normalize=True)
        
        data = (self.cm.chunk({'time': 10})/(24*3600)).load()
        cm_psd = mean_rapsd(data[:,:,offset:num_latitudes+offset], normalize=True)

        return psd_scale, era5_psd, esm_psd, sde_psd, cm_psd 


    def plot_power_spectral_densities(self):

        fname = self.out_path + 'psd.pdf'

        fig = plt.figure(figsize=(7,9), constrained_layout=True)
        plt.tick_params(width=1.5)
        mpl.rcParams['axes.linewidth'] = 1.5

        subfig = fig.subfigures(nrows=1, ncols=1)
        axes = subfig.subplots(nrows=2, ncols=1)

        ax1 = axes[0]

        ax1.annotate("A", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax1,
                         bbox=None) 

        # plot 1 - ERA5
        cfg = self.plot_config['era5']
        ax1.plot(1/self.era5_psd[1]*0.75*111/2, self.era5_psd[0], label=cfg['label'], color=cfg['color'], lw=cfg['lw'], alpha=cfg['alpha'])

        # plot 2 - ESM
        cfg = self.plot_config['esm']
        ax1.plot(1/self.esm_psd[1]*0.75*111/2, self.esm_psd[0], label=cfg['label'], color=cfg['color'], lw=cfg['lw'], alpha=cfg['alpha'])

        # plot 3 - SDE
        cfg = self.plot_config['sde']
        ax1.plot(1/self.sde_psd[1]*0.75*111/2, self.sde_psd[0], label=cfg['label'], color=cfg['color'], lw=cfg['lw'], alpha=cfg['alpha'])

        # plot 4 - CM
        cfg = self.plot_config['cm']
        cfg['label'] = 'CM'
        ax1.plot(1/self.cm_psd[1]*0.75*111/2, self.cm_psd[0], label=cfg['label'], color=cfg['color'], lw=cfg['lw'], alpha=cfg['alpha'])

        freq_min = 0.055
        ax1.axvline(x=1/freq_min*0.75*111/2, c='tab:gray', ls='--', lw=2, label=f'Intersection')

        # options
        ax1.set_yscale('log', base=2)
        ax1.set_ylabel("Power spectral density [a.u.]")
        ax1.tick_params(width=1.5)
        ax1.set_xscale('log', base=2)
        ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax1.set_xlim(64, 16000)
        ax1.legend(frameon=False, loc='center right').get_frame().set_boxstyle('Square')

        # scale analysis:

        ax1 = axes[1]

        ax1.annotate("B", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax1,
                         bbox=None) 

        # plot 1 - ERA5
        cfg = self.plot_config['era5']
        ax1.plot(1/self.era5_psd[1]*0.75*111/2, self.era5_psd[0], label=cfg['label'], color=cfg['color'], alpha=1, lw=3)

        # plot 2 - ESM
        cfg = self.plot_config['esm']
        cax = ax1.plot(1/self.esm_psd[1]*0.75*111/2, self.esm_psd[0], label=cfg['label'], color=cfg['color'], alpha=1, lw=3.5)

        time_list_reduced = self.time_list
        colors = plt.cm.viridis(np.linspace(0,1,len(time_list_reduced)))

        cfg = self.plot_config['cm']
        for i, t in enumerate(time_list_reduced):
            ax1.plot(1/self.psd_scale[t][1]*0.75*111/2, self.psd_scale[t][0], color=colors[i], lw=cfg['lw'], alpha=0.6, ls='--')

        ax1.set_yscale('log', base=2)
        ax1.set_xscale('log', base=2)
        ax1.set_ylabel("Power spectral density [a.u.]")
        ax1.tick_params(width=1.5)
        ax1.set_xlabel("Wavelength [km]")
        ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax1.set_xlim(64, 16000)

        norm = mpl.colors.Normalize(vmin=0., vmax=80)

        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), aspect=40, label=r'Noise strength $t$')

        plt.savefig(fname, format='pdf', bbox_inches='tight')
        print(fname)


    def plot_biases(self):

        plot_bias_config = {
                            'era5': {'label': 'ERA5', 'color': 'black', 'lw': 2.5, 'alpha': 0.8},
                            'esm_lr': {'label': 'POEM (raw)', 'color': 'grey', 'lw': 2.5, 'alpha': 0.8},
                            'esm': {'label': 'POEM', 'color': 'orange', 'lw': 2.5, 'alpha': 0.8},
                            'sde': {'label': 'SDE', 'color': 'tab:cyan', 'lw': 2.5, 'alpha': 0.8},
                            'sde_lr': {'label': 'SDE (upscaled)', 'color': 'tab:cyan', 'lw': 2.5, 'alpha': 0.8},
                            'cm': {'label': 'CM', 'color': 'magenta', 'lw': 2.5, 'alpha': 0.8},
                            'cm_lr': {'label': 'CM (upscaled)', 'color': 'magenta', 'lw': 2.5, 'alpha': 0.8},
                            }

        fname = self.out_path + 'distributional_bias.pdf'

        # histograms 

        fig = plt.figure(figsize=(9,7.5), constrained_layout=True)
        plt.tick_params(width=1.5)
        mpl.rcParams['axes.linewidth'] = 1.5
        lw = 1.6

        subfig = fig.subfigures(nrows=1, ncols=1)
        axes = subfig.subplots(nrows=2, ncols=2)

        ax = axes[0,0]

        ax.annotate("A", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 


        cfg = plot_bias_config['era5']
        data = self.era5
        r = ax.hist(data.values.flatten(), bins=500, density=True, histtype='step',
                     label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['esm_lr']
        data = self.esm_hr_no_qm
        r = ax.hist(data.values.flatten(), bins=500, density=True, histtype='step',
                     label='POEM (without QDM)', color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['esm']
        data = self.esm_hr
        r = ax.hist(data.values.flatten(), bins=500, density=True, histtype='step',
                     label='POEM (with QDM', color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['sde']
        data = self.sde
        r = ax.hist(data.values.flatten(), bins=500, density=True, histtype='step',
                     label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['cm']
        data = self.cm
        r = ax.hist(data.values.flatten(), bins=500, density=True, histtype='step',
                     label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        ax.set_ylim(5e-8, 1.1)
        ax.set_xlim(0, 165)
        ax.set_yscale('log')
        ax.set_ylabel("Histogram")
        ax.set_xlabel("Precipitation [mm/day]")

        # absolute error 

        ax = axes[0,1]

        ax.annotate("B", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 

        cfg = plot_bias_config['era5']
        data = self.era5
        era5_hist = np.histogram(data.values.flatten(), bins=500,  density=True)

        cfg = plot_bias_config['esm_lr']
        data = self.esm_hr_no_qm
        data_hist = np.histogram(data.values.flatten(), bins=500,  density=True)
        ax.plot(era5_hist[1][1:], abs(era5_hist[0]-data_hist[0]),
                 label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['esm']
        data = self.esm_hr
        data_hist = np.histogram(data.values.flatten(), bins=500,  density=True)
        ax.plot(era5_hist[1][1:], abs(era5_hist[0]-data_hist[0]),
                 label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['sde']
        data = self.sde
        data_hist = np.histogram(data.values.flatten(), bins=500,  density=True)
        ax.plot(era5_hist[1][1:], abs(era5_hist[0]-data_hist[0]),
                 label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['cm']
        data = self.cm
        data_hist = np.histogram(data.values.flatten(), bins=500,  density=True)
        ax.plot(era5_hist[1][1:], abs(era5_hist[0]-data_hist[0]),
                 label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        ax.set_ylim(1e-7, 5)
        ax.set_xlim(0, 165)

        ax.set_yscale('log')
        ax.set_ylabel("Absolute error")
        ax.set_xlabel("Precipitation [mm/day]")

        # latitude profile 

        ax = axes[1,0]
        ax.annotate("C", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 

        latitudes = self.era5.latitude

        cfg = plot_bias_config['era5']
        data = self.era5.mean(dim=("time", "longitude"))
        l_era5, = ax.plot(latitudes, data, label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['esm_lr']
        data = self.esm_hr_no_qm.mean(dim=("time", "longitude"))
        l_esm_no_qm, = ax.plot(latitudes, data, label='POEM (without QDM)', color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['esm']
        data = self.esm_hr.mean(dim=("time", "longitude"))
        l_esm, = ax.plot(latitudes, data, label='POEM (with QDM)', color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['sde']
        data = self.sde.mean(dim=("time", "longitude"))
        l_sde, = ax.plot(latitudes, data, label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['cm']
        data = self.cm.mean(dim=("time", "longitude"))
        l_cm, = ax.plot(latitudes, data, label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        ax.set_xlabel(r"Latitude [$^{\circ}$N]")
        ax.set_ylabel("Mean precipitation [mm/day]")

        ax.set_xlim(-95, 95)

        # absolute error

        ax = axes[1,1]

        ax.annotate("D", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 

        latitudes = self.era5.latitude

        cfg = plot_bias_config['era5']
        era5_profile = self.era5.mean(dim=("time", "longitude"))

        cfg = plot_bias_config['esm_lr']
        data = self.esm_hr_no_qm.mean(dim=("time", "longitude"))
        rel_error = abs(era5_profile - data)
        ax.plot(latitudes, rel_error, label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['esm']
        data = self.esm_hr.mean(dim=("time", "longitude"))
        rel_error = abs(era5_profile - data)
        ax.plot(latitudes, rel_error, label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['sde']
        data = self.sde.mean(dim=("time", "longitude"))
        rel_error = abs(era5_profile - data)
        ax.plot(latitudes, rel_error, label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        cfg = plot_bias_config['cm']
        data = self.cm.mean(dim=("time", "longitude"))
        rel_error = abs(era5_profile - data)
        l1, = ax.plot(latitudes, rel_error, label=cfg['label'], color=cfg['color'], lw=lw, alpha=cfg['alpha'])

        ax.set_xlabel(r"Latitude [$^{\circ}$N]")
        ax.set_ylabel("Absolute error [mm/day]")
        ax.set_ylim(-0.1, 2.5)

        lgnd = fig.legend(loc='upper center', handles=[l_era5, l_esm_no_qm, l_esm, l_sde, l_cm], bbox_to_anchor=(0.5, 0),
                  fancybox=False, shadow=False, ncol=5, edgecolor='k').get_frame().set_boxstyle('Square')


        plt.savefig(fname, format='pdf', bbox_inches='tight')
        print(fname)


    def plot_uncertainty_quantification(self):

        mpl.rcParams['axes.linewidth'] = .5

        path = '/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge45tac2/diffusion-downscaling/results/figures/'
        fname = path + 'uncertainty_quantification.png'

        fig = plt.figure(figsize=(9,10), constrained_layout=True)

        subfig = fig.subfigures(nrows=1, ncols=1)
        axes = subfig.subplots(nrows=3, ncols=2, sharey=True)

        # esm field
        ax = axes[0, 0]
        ax.annotate("A", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        cfg = self.plot_config['cm']
        cfg['label'] = "POEM (interpolated to "+r"$4 \times r_{\mathrm{POEM}} $"+")" 

        data = self.esm_hr[0]
        lats = data.latitude
        lons = data.longitude
        self.single_field(data, lats, lons, ax, cfg, vmin=0.1, vmax=20)

        ax = axes[0, 1]
        ax.annotate("A", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 


        cfg = self.plot_config['cm']
        cfg['label'] = "POEM (interpolated to "+r"$4 \times r_{\mathrm{POEM}} $"+")" 

        data = self.esm_hr[0]
        lats = data.latitude
        lons = data.longitude
        self.single_field(data, lats, lons, ax, cfg, vmin=0.1, vmax=20)

        # single sample

        ax = axes[1, 0]
        ax.annotate("B", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        cfg = self.plot_config['cm']
        cfg['label'] = "CM single sample ("+r"$r = 4 \times r_{\mathrm{POEM}} $"+")" 
        data = self.cm_ens.isel(samples=0)[0]
        lats = data.latitude
        lons = data.longitude
        self.single_field(data, lats, lons, ax, cfg, vmin=0.1, vmax=20)

        ax = axes[1, 1]
        ax.annotate("C", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        cfg['label'] = "CM single sample ("+r"$r = 4 \times r_{\mathrm{POEM}} $"+")" 
        cfg = self.plot_config['cm']
        data = self.cm_ens.isel(samples=1)[0]
        lats = data.latitude
        lons = data.longitude
        cs = self.single_field(data, lats, lons, ax, cfg, vmin=0.1, vmax=20)

        cbar = subfig.colorbar(cs, ax=axes[:2,:], location='bottom', shrink=0.25, extend='max', label="Precipitation [mm/day]")
        cbar.set_ticks(ticks=[1,5,10,15,20])

        # sample mean

        ax = axes[2, 0]
        ax.annotate("D", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        cfg = self.plot_config['cm']
        cfg['label'] = "CM sample mean" 
        data = self.cm_ens.mean(dim=("samples", "time"))
        lats = data.latitude
        lons = data.longitude
        cs = self.single_field(data, lats, lons, ax, cfg, vmin=0.0, vmax=10, plot_mask=False)

        cbar = subfig.colorbar(cs, ax=axes[2,0], location='bottom', shrink=0.5, extend='max', label="Mean [mm/day]")

        # sample spread

        ax = axes[2, 1]
        ax.annotate("E", ha="center", va="center", size=14,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
        cfg = self.plot_config['cm']
        cfg['label'] = "CM sample spread (SD)" 
        data = self.cm_ens.std(dim=("samples", "time"))
        lats = data.latitude
        lons = data.longitude
        cs = self.single_field(data, lats, lons, ax, cfg, vmin=0.0, vmax=10, plot_mask=False, cmap='viridis')

        cbar = subfig.colorbar(cs, ax=axes[2,1], location='bottom', shrink=0.5, extend='max', label="Standard deviation (SD) [mm/day]")
        plt.savefig(fname, format='png', bbox_inches='tight', dpi=400)
        print(fname)


    def compute_trends(self):

        glob_mean_delayed = self.cm_trend.mean(dim=("latitude", "longitude")).resample(time="M").mean().rolling(time=52*3, center=True).mean().dropna("time")
        with ProgressBar():
            self.glob_mean_cm = glob_mean_delayed.compute()

        glob_mean_delayed = self.dqm_ssp.mean(dim=("latitude", "longitude")).resample(time="M").mean().rolling(time=52*3, center=True).mean().dropna("time")
        with ProgressBar():
            self.glob_mean_dqm = glob_mean_delayed.compute()


    def plot_trends(self):

        plot_trend_config = {
                        'esm': {'label': 'POEM', 'color': 'orange', 'lw': 2.5, 'alpha': 0.8},
                        'cm': {'label': 'CM', 'color': 'magenta', 'lw': 2.5, 'alpha': 0.8},
                        }

        fig_fname = self.out_path + 'figures/future_trends_ssp585_f.png'

        plt.figure(figsize=(8,4.25))
        mpl.rcParams['axes.linewidth'] = 1.5
        plt.tick_params(width=1.5)
        time = self.glob_mean_dqm.time
        ax = plt.subplot(1,1,1)

        cfg = plot_trend_config['esm']
        data = (self.glob_mean_dqm.precipitation - self.glob_mean_dqm.precipitation[0])
        plt.plot(time, data, label=cfg['label']+" SSP5-8.5", color=cfg['color'], lw=cfg['lw'], alpha=cfg['alpha'])

        cfg = plot_trend_config['cm']
        data = (self.glob_mean_cm.precipitation - self.glob_mean_cm.precipitation[0])
        plt.plot(time, data, label=cfg['label'], color=cfg['color'], lw=cfg['lw'], alpha=cfg['alpha'])

        plt.xlabel(r"Time [monthly]")
        plt.ylabel("Global mean precipitation [mm/month]")

        ax.annotate(f"F", ha="center", va="center", size=14,
                             xy=(1-0.955, 0.925), xycoords=ax,
                             bbox=dict( fc="white", ec="w", lw=1)) 

        plt.legend(frameon=False, loc='center left').get_frame().set_boxstyle('Square')

        plt.savefig(fig_fname, format='png', dpi=350, bbox_inches='tight')
        print(fig_fname)