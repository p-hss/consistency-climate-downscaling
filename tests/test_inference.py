import unittest
import math
import torch

from src.configuration import Config
from src.sde_model.model import SDEModel
from src.sde_model.inference import Inference


class TestInference(unittest.TestCase):
    """Runs some basic tests for the inference class. """


    def test_unconditional_sampling(self):

        config = Config()

        for channels in range(1,2):

            config.in_channels = channels
            config.out_channels = 1
            config.num_batches = 1
            config.batch_size = 1
            config.sample_dimension = (32,32)
            config.show_valid_samples_tensorboard = False

            inf = Inference(config)

            model = SDEModel(config) # initialize new network

            inf.model= model

            y = inf.run(sampler_type="sde",
                        convert_to_xarray=False,
                        inverse_transform=False)

            self.assertFalse(math.isnan(y.mean()), "sample is NaN")
            self.assertEqual(y.shape, (config.num_batches,
                                       config.out_channels,
                                       config.sample_dimension[0],
                                       config.sample_dimension[1]), "incorrect output shape")


if __name__ == "__main__":
    unittest.main()