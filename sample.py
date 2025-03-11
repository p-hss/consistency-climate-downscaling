import xarray as xr
from src.configuration import parse_command_line
from src.sde_model.inference import Inference as SDEInference
from src.consistency_model.inference import ConsistencyInference as CMInference


def main():
    """ Command line interface for generating samples. """

    config = parse_command_line()
    config.checkpoint_path = '/results'
    config.sample_dimension = [64,96]
    
    if config.diffusion_model == 've':
        inf = SDEInference(config)
    elif config.diffusion_model == 'consistency':
        inf = CMInference(config)

    # for converting back to xarray DataArray:
    inf.training_target = inf.test_input = xr.open_dataset('/data/datasets/era5_3deg.nc')
    inf.load_model(checkpoint_fname=f'best_{config.diffusion_model}_model.ckpt')
    samples = inf.run(convert_to_xarray=True,
                      inverse_transform=True)

    print(samples)


if __name__ == "__main__":
    main()