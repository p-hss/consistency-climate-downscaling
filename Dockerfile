# hash:sha256:8985eb054004f4a4a9a0b6e0e81ff65937c8af12b65d2ae07a1ee81f830a5071
FROM registry.codeocean.com/codeocean/pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults pip && conda clean -ya

RUN pip install -U --no-cache-dir \
    pytorch-lightning \
    numpy \
    xarray \
    dask \
    netcdf4 \
    jupyter\
    jupyterlab \
    matplotlib \
    basemap \
    torchvision \
    diffusers==0.14.0 \
    tensorboard \
    matplotlib \
    scipy \
    tqdm

