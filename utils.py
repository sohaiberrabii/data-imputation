import os
from typing import Iterable

from tqdm import tqdm
from typing import Optional
from numpy import ndarray
import numpy as np
from urllib.request import urlopen
from zipfile import ZipFile


def download_and_unzip(
        url: str,
        extract_to: str = '.',
        filename: Optional[str] = None,
        remove_after: bool = False) -> None:
    if not filename:
        filename = os.path.join(
                extract_to,
                os.path.basename(url)
        )

    _urlretrieve(url, filename)
    _unzip(filename, extract_to, remove_after=remove_after)


def ampute(
        data: ndarray,
        miss_rate: float,
        missingness: str = "sequence",
        pvals: Optional[Iterable] = None) -> ndarray:
    """Generate missing data depending on the given mechanism.

    Args:
        data (ndarray): The data array.
        miss_rate (float): The rate of missingness
        missingness (str): Can be "mcar" for missing completely at random or
            "sequence" for random sequences of missingness, simulating
            failure in time series.
        pvals (iterable, optional): Only used when ``missingness`` is
            "sequence". Gives the probabilities of occurrence
            (or increase in length) for different sequences. Must sum to one.
    """
    if missingness == "mcar":
        return _ampute_mcar(data, miss_rate)
    elif missingness == "sequence":
        return _ampute_multinomial(data, miss_rate, pvals)
    else:
        raise NotImplemented("missingness mechanism must be mcar or sequence")


def _ampute_multinomial(
        data: ndarray,
        miss_rate: float, pvals: Optional[Iterable] = None) -> ndarray:
    """Generate missing data for time series.

    Args:
        data (ndarray): The complete data in shape :math:`(N, L, H)`. Where
            ``L`` is the sequence length (or number of timesteps).
        miss_rate (float): Probability of missingness.
        pvals (iterable, optional): Sequences of missing data are generated
            according to these probabilities, bias implies longer sequences.

    Returns:
        data_miss (ndarray): data with missing values as nan.
    """
    if pvals is None:
        pvals = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
    n_chunks = len(pvals)
    n_samples, n_timesteps, n_features = data.shape

    # Generate random lengths of missing chunks that sum to required miss_rate
    miss_total = int(n_timesteps * miss_rate)
    missing_lengths = np.random.multinomial(
        miss_total,
        pvals,
        size=(n_samples, n_features),
    )

    # Generate random lengths of non-missing chunks
    retained_lengths = np.random.multinomial(
        n_timesteps - miss_total,
        np.ones(n_chunks + 1) / (n_chunks + 1),
        size=(n_samples, n_features),
    )

    # Interleave the missing and non-missing chunks
    interleaved_chunks = np.dstack(
        [chunks for i in range(n_chunks)
         for chunks in (retained_lengths[:, :, i], missing_lengths[:, :, i])]
        + [retained_lengths[:, :, -1]]
    )

    # Ampute the data  (TODO: Can this be vectorized ?)
    idx = np.cumsum(interleaved_chunks, axis=2)
    data_miss = data.copy()
    for k in range(interleaved_chunks.shape[-1] - 1):
        for i in range(n_samples):
            for j in range(n_features):
                if k % 2 == 0:
                    data_miss[i, idx[i, j, k]:idx[i, j, k + 1], j] = np.nan

    return data_miss


def _ampute_mcar(data: ndarray, miss_rate: float) -> ndarray:
    """Generate MCAR missing data.

    Args:
        data (ndarray): the complete data.
        miss_rate (float): probability of missingness.

    Returns:
        data_miss (ndarray): data with missing values as nan.
    """
    mask = np.random.choice([0, 1], data.shape, p=[miss_rate, 1 - miss_rate])
    data_miss = data.copy()
    data_miss[mask == 0] = np.nan
    return data_miss


def imputation_rmse(x: ndarray, x_hat: ndarray, mask: ndarray) -> float:
    """Compute the rmse between observed and imputed data.

    Args:
        x (ndarray): The complete data.
        x_hat (ndarray): The imputed data.
        mask (ndarray): Binary mask, 1 means observed and 0 missing/imputed.

    Returns:
        float: the root mean square error.
    """
    mse = np.sum(((1 - mask) * x - (1 - mask) * x_hat) ** 2) / np.sum(1 - mask)
    return np.sqrt(mse)


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urlopen(url) as response:
            with tqdm(total=response.length, desc='Downloading') as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(chunk_size)


def _unzip(from_path: str, to_path: str, remove_after: bool = False) -> None:
    zipfile = ZipFile(from_path)
    for zipinfo in tqdm(zipfile.infolist(), desc='Extracting'):
        if not zipinfo.filename.startswith('__MACOSX'):
            zipinfo.filename = zipinfo.filename.replace(" ", "")
            zipfile.extract(zipinfo, path=to_path)

    if remove_after:
        os.remove(from_path)


if __name__ == '__main__':
    _ampute_multinomial(np.random.rand(7000, 128, 9), miss_rate=0.2)
