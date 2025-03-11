import numpy as np
import xarray as xr
from tqdm import tqdm


def mean_rapsd(data: xr.DataArray, normalize: bool=False):
    """
    Averages the RAPSD in time over a DataArray.

    Args:
        data: The dataset with shape [time, latitude, longitude]
        normalize: Normalize the spectra
    
    Returns:
        Average RAPSD, Fourier frequencies
    """

    assert(len(data.latitude) == len(data.longitude)), "Number of latitude coordinates must equal the number of longitudes."

    mean_psd = np.zeros(len(data.latitude)//2)

    for i in tqdm(range(len(data))):
        data_slice = data[i].values
        psd, freq = rapsd(data_slice, fft_method=np.fft, normalize=normalize, return_freq=True)
        mean_psd += psd

    mean_psd /= len(data.time)

    return mean_psd, freq


def mean_rapsd_numpy(data: np.ndarray, normalize: bool=False):
    """
    Averages the RAPSD in time over a DataArray.

    Args:
        data: The dataset with shape [time, latitude, longitude]
        normalize: Normalize the spectra
    
    Returns:
        Average RAPSD, Fourier frequencies
    """

    mean_psd = np.zeros(data.shape[-2]//2)

    for t in tqdm(range(len(data))):
        data_slice = data[t]
        psd, freq = rapsd(data_slice, fft_method=np.fft, normalize=normalize, return_freq=True)
        mean_psd += psd

    mean_psd /= len(data)

    return mean_psd, freq


def rapsd(field: np.ndarray,
          fft_method=np.fft,
          return_freq: bool=False,
          d: float=1.0,
          normalize: bool=False
          ) -> np.ndarray:
    """

    Adapted from https://github.com/pySTEPS/pysteps/blob/57ece4335acffb111d4de7665fb678b875d844ac/pysteps/utils/spectral.py#L100

    Compute radially averaged power spectral density (RAPSD) from the given
    2D input field.

    Args: 
        field: A 2d array of shape (m, n) containing the input field.
        fft_method: A module or object implementing the same methods as numpy.fft and
            scipy.fftpack. If set to None, field is assumed to represent the
            shifted discrete Fourier transform of the input field, where the
            origin is at the center of the array
            (see numpy.fft.fftshift or scipy.fftpack.fftshift).
        return_freq: Whether to also return the Fourier frequencies.
        d: Sample spacing (inverse of the sampling rate). Defaults to 1.
            Applicable if return_freq is 'True'.
        normalize: If True, normalize the power spectrum so that it sums to one.

    Returns:
    out: One-dimensional array containing the RAPSD. The length of the array is
         int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
    freq: One-dimensional array containing the Fourier frequencies.
    """

    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number "
            "of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    yc, xc = compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(field.shape[0], field.shape[1])

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result


def compute_centred_coord_array(M: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Compute a 2D coordinate array, where the origin is at the center.

    Args: 
        M: The height of the array.
        N: The width of the array.

    Returns:
        The coordinate array.
    """

    if M % 2 == 1:
        s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2) : int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2) : int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC

