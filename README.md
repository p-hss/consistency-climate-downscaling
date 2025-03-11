# Generative climate model downscaling

## Overview

This is the repository of the paper "Fast, scale-adaptive and uncertainty-aware downscaling of Earth system model fields with generative machine learning" (https://www.nature.com/articles/s42256-025-00980-5).

## Usage
Set paths to training data and hyperparameters in `src/configuration.py`. Some can be changed via the shell when executing the training run with:

```bash
python main.py \
    --name "my_model" \
    --n_epochs 50 \
    --batch_size 8 
```

## Test

Run unit tests with 

```bash
 python -m unittest discover tests/
```

## References
The implementation is mostly based on the repositories:

- https://github.com/yang-song/score_sde_pytorch 
- https://github.com/openai/consistency_models.

as well as the papers:

- Y. Song et at., 2021: https://arxiv.org/abs/2011.13456
- Y. Song et at., 2023: https://arxiv.org/abs/2303.01469
- Bischoff and Deck, 2023: https://arxiv.org/abs/2305.01822
- T. Karras et al., 2022: https://arxiv.org/abs/2206.00364

