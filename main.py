from src.configuration import parse_command_line
from src.training import training
from src.sde_model.model import SDEModel
from src.utils.utils import get_checkpoint_path, create_paths
from src.consistency_model.model import Consistency

def main():
    """ Main executable to start training from command line. """

    config = parse_command_line()

    create_paths(config)
    get_checkpoint_path(config)

    if config.diffusion_model == 'consistency':
        model = Consistency(config)

    else:
        model = SDEModel(config)

    training(config, model, verbose=False)

    print("Training finished.")


if __name__ == "__main__":
    main()