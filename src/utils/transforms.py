import numpy as np
import xarray as xr


def apply_transforms(data: xr.DataArray,
                     data_ref: xr.DataArray,
                     config) -> xr.DataArray:
    """ Apply a sequence of transformations given a training set reference

    Args:
        data: Data to be transformed
        data_ref: Reference data from the training set
        config: Conifguration dataclass of transforms and constants

    Returns:
        The transformed data
    
    """

    if 'log' in config.transforms:
        data = log_transform(data, config.epsilon)
        data_ref = log_transform(data_ref, config.epsilon)

    if 'standardize' in config.transforms:
        data = standardize(data, data_ref)
        data_ref = standardize(data_ref, data_ref)

    if 'normalize' in config.transforms:
        data = norm_transform(data, data_ref)

    if 'normalize_minus1_to_plus1' in config.transforms:
        data = norm_minus1_to_plus1_transform(data, data_ref)
        
    return data   


def apply_inverse_transforms(data: xr.DataArray,
                            data_ref: xr.DataArray,
                            config) -> xr.DataArray:
    """ Apply a sequence of inverse transformations given a training set reference

    Args:
        data: Data to be transformed
        data_ref: Reference data from the training set
        config: Conifguration dataclass of transforms and constants
    
    Returns:
        The data tranformed back to the physical space
    """

    if "log" in config.transforms:
        data_ref = log_transform(data_ref, config.epsilon)

    if "standardize" in config.transforms:
        data_ref_ = standardize(data_ref, data_ref) 

    if "normalize_minus1_to_plus1" in config.transforms:
        if "standardize" in config.transforms:
            data = inv_norm_minus1_to_plus1_transform(data, data_ref_)
        else:
            data = inv_norm_minus1_to_plus1_transform(data, data_ref)

    if "standardize" in config.transforms:
        data = inv_standardize(data, data_ref)

    if "log" in config.transforms:
        data = inv_log_transform(data, config.epsilon)

    return data


def log_transform(x, epsilon):
    return np.log(x + epsilon) - np.log(epsilon)


def inv_log_transform(x, epsilon):
    return np.exp(x + np.log(epsilon)) - epsilon


def standardize(x, x_ref):
    return (x - x_ref.mean(dim='time'))/x_ref.std(dim='time')


def inv_standardize(x, x_ref):
    x = x*x_ref.std(dim='time')
    x = x + x_ref.mean(dim='time')
    return x


def norm_transform(x, x_ref):
    return (x - x_ref.min(dim='time'))/(x_ref.max(dim='time') - x_ref.min(dim='time'))


def inv_norm_transform(x, x_ref):
    return x * (x_ref.max(dim='time') - x_ref.min(dim='time')) + x_ref.min(dim='time')


def norm_minus1_to_plus1_transform(x, x_ref, use_quantiles=False, q_max=0.999):
    if use_quantiles: 
        x = (x - x_ref.quantile(1-q_max,dim='time'))/(x_ref.quantile(q_max,dim='time') - x_ref.quantile(1-q_max,dim='time'))
    else:
        x = (x - x_ref.min())/(x_ref.max() - x_ref.min())
    x = x*2 - 1
    return x 


def inv_norm_minus1_to_plus1_transform(x, x_ref, use_quantiles=False, q_max=0.999):
    x = (x + 1)/2
    if use_quantiles: 
        x = x * (x_ref.quantile(q_max) - x_ref.quantile(1-q_max)) + x_ref.quantile(1-q_max)
    else:
        x = x * (x_ref.max() - x_ref.min()) + x_ref.min()
    return x


