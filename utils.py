import os
from tqdm import tqdm
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
    for zipinfo in tqdm(zipfile.infolist()[1:], desc='Extracting'):
        if not zipinfo.filename.startswith('__MACOSX'):
            zipinfo.filename = os.path.join(
                *(zipinfo.filename.split(os.path.sep)[1:])
            ).replace(" ", "")
            zipfile.extract(zipinfo, path=to_path)

    if remove_after:
        os.remove(from_path)


if __name__ == '__main__':
    # UCI-HAR Dataset
    os.makedirs(os.path.join('data', 'UCIHARDataset'), exist_ok=True)
    download_and_unzip(
        'https://archive.ics.uci.edu/ml/machine-learning'
        '-databases/00240/UCI%20HAR%20Dataset.zip',
        extract_to='UCIHARDataset'
    )
