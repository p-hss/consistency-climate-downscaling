import unittest
import math
import numpy as np
import xarray as xr

from src.configuration import Config
from src.utils.transforms import apply_transforms, apply_inverse_transforms


class TestData(unittest.TestCase):
    """Runs some basic tests on the training data. """

    def test_transforms(self):

        config = Config()
        config.transforms = ["log", "normalize_minus1_to_plus1"]

        x = xr.DataArray(np.random.rand(1,32,32), 
                         coords={
                                'time': np.random.rand(1),
                                'latitude':  np.arange(32),
                                'longitude': np.arange(32)}, 
                         dims=["time", "latitude", "longitude"])

        x_ref = xr.DataArray(np.random.rand(1,32,32) + 5, 
                         coords={
                                'time': np.random.rand(1),
                                'latitude':  np.arange(32),
                                'longitude': np.arange(32)}, 
                         dims=["time", "latitude", "longitude"])

        x_tranformed = apply_transforms(x, x_ref, config)
        x_inv_tranformed = apply_inverse_transforms(x_tranformed, x_ref, config)

        self.assertTrue(math.isclose(x.mean(), x_inv_tranformed.mean()),
                        "inverse transform error")

        self.assertTrue(math.isclose(x.std(), x_inv_tranformed.std()),
                        "inverse transform error")


if __name__ == "__main__":
    unittest.main()