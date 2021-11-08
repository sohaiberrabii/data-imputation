import os
from tqdm import tqdm
from typing import Tuple
import numpy as np
from urllib.request import urlopen
from zipfile import ZipFile


def download_and_unzip(
        url: str, extract_to: str = '.', remove_after: bool = False) -> None:
    filename = extract_to + ".zip"
    _urlretrieve(url, filename)
    _unzip(filename, extract_to, remove_after=remove_after)


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
        path_parts = zipinfo.filename.split(os.path.sep)
        if not zipinfo.filename.startswith('__MACOSX') \
                and len(path_parts) > 1 and path_parts[1]:
            zipinfo.filename = os.path.join(*(path_parts[1:])).replace(" ", "")
            zipfile.extract(zipinfo, path=to_path)

    if remove_after:
        os.remove(from_path)


def binary_sampler(p: float, shape: Tuple[int, ...]) -> np.ndarray:
    """Sample random binary variables.

    Args:
        p (float): probability of 1.
        shape (tuple): shape of the generated binary array.

    Returns:
        ndarray: generated random binary array.
    """
    return np.random.choice([0, 1], size=shape, p=[1 - p, p])


def ampute(data: np.ndarray, miss_rate: float) -> np.ndarray:
    """Generate missing data.

    Args:
        data (ndarray): the complete data.
        miss_rate (float): probability of missingness.

    Returns:
        data_miss (ndarray): data with missing values as nan.
    """
    mask = binary_sampler(1 - miss_rate, data.shape)
    data_miss = data.copy()
    data_miss[mask == 0] = np.nan
    return data_miss
