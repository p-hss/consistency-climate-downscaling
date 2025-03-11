from typing import List, Optional, Type, Union, Tuple
import argparse
import xarray as xr
import numpy as np
from tqdm import tqdm
from xclim import sdba 

import src.utils.xarray_utils as xu


class QuantileMapping():

    def __init__(self,
                 model_path: str,
                 target_path: str,
                 out_path: str,
                 num_quantiles: Optional[int]=100,
                 train_set: Optional[Tuple[str,str]]=['1950', '2000'],
                 test_set: Optional[Tuple[str,str]]=['2020', '2100'],
                 verbose: Optional[bool]=True
                 ):
        """Peforms quantile mapping on model data given a target dataset.

        Args:
            model_path: Path to simlulatin data in .nc format.
            target_path: Path to target data in .nc format.
            out_path: Path where to store the result.
            num_quantiles: Number of quantiles used for the mapping. 
            train_set: Beginnig and end of training period. 
            verbose: Enables verbose printing.
        """

        self.verbose = verbose

        self.model_path = model_path
        if self.verbose: print(model_path)
        
        self.target_path = target_path
        if self.verbose: print(target_path)

        self.out_path = out_path
        if self.verbose: print(out_path)
        
        self.train_set = train_set
        self.test_set = test_set
        self.num_quantiles = num_quantiles


    def load_data(self):
        """Loads the data from file. """
        print("test period", self.test_set)

        if self.verbose: print('loading data..')
        model = xr.open_dataset(self.model_path,
                                #chunks={'time': 50}) \
                                chunks=None) \
                                    .precipitation \
                                    .astype(np.float32).load()

        self.model_simulation = model.sel(time=slice(self.test_set[0], self.test_set[1])).convert_calendar('noleap')
 
        self.model_historical = model.sel(time=slice(self.train_set[0], self.train_set[1])).convert_calendar('noleap')

        if model.longitude[0] > 0:
            self.model_historical = xu.shift_longitudes(self.model_historical)
            self.model_simulation = xu.shift_longitudes(self.model_simulation)

        self.target = xr.open_dataset(self.target_path,
                                      #chunks={'time': 50} ) \
                                      chunks=None ) \
                                      .precipitation \
                                      .astype(np.float32).load()

        self.target_historical = self.target.sel(time=slice(self.train_set[0], self.train_set[1])).convert_calendar('noleap')
        
        self.target_historical = xu.remove_leap_year(self.target_historical)
        self.target_historical["time"] = self.model_historical.time

        if self.verbose: print('finished.')


    def run(self):
        """Peforms the quantile mapping. """
        
        if self.verbose: print('fitting quantiles..')
        group = sdba.adjustment.Grouper("time")
        
        self.result = xr.zeros_like(self.model_simulation)
        
        num_latitude  = len(self.target.latitude)
        num_longitude = len(self.target.longitude)

        print("test", len(self.model_simulation))
        
        for lat in tqdm(range(num_latitude)):
            for lon in tqdm(range(num_longitude), leave=False):
                
                qm = sdba.adjustment.QuantileDeltaMapping

                target = self.target_historical.isel(latitude=lat, longitude=lon)
                model_historical = self.model_historical.isel(latitude=lat, longitude=lon)
                model_simulation = self.model_simulation.isel(latitude=lat, longitude=lon)

                Adj = qm.train(target, model_historical,
                               nquantiles=self.num_quantiles, 
                               group=group,
                               skip_input_checks=True)
                
                mapped = Adj.adjust(sim=model_simulation, skip_input_checks=True)
                
                if mapped.min() < 0:
                    print(f"negative values in qm result, lat={lat}, lon={lon}, min = {mapped.min().values}")
                    mapped = xr.where(mapped < 0, 0, mapped, keep_attrs=True)

                self.result[:, lat, lon] = mapped

        if self.verbose: print('finished.')


    def save(self):
        xu.write_dataset(self.result, self.out_path)


def parse_command_line():
    """ Parses the command line options. """

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path",
                        help="Path to the model .nc file", type=str)

    parser.add_argument("-t", "--target_path",
                        help="Path to the target .nc file", type=str)

    parser.add_argument("-o", "--out_path",
                        help="Path to the output .nc file", type=str)

    parser.add_argument("-ts", "--training_start",
                        help="Start year of training data", type=int)

    parser.add_argument("-te", "--training_end",
                        help="Start year of training data", type=int)

    parser.add_argument("-nq", "--num_quantiles",
                        help="Number of quantiles", type=int)

    parser.add_argument("-v", "--verbose",
                        help="Verbose output", action='store_true')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_command_line()

    qm = QuantileMapping(model_path=args.model_path,
                         target_path=args.target_path,
                         out_path=args.out_path,
                         num_quantiles=args.num_quantiles,
                         train_set=[str(args.training_start), str(args.training_end)],
                         verbose=args.verbose)
    qm.load_data()
    qm.run()
    qm.save()