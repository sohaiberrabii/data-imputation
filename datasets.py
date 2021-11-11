import os
from typing import List, Tuple, Any
from torch import Tensor
from torch.utils.data import Dataset
from utils import download_and_unzip, ampute
from sklearn.preprocessing import MinMaxScaler
from urllib.error import URLError
import pandas as pd
import numpy as np


class UCIHAR(Dataset):
    """`UCI-HAR <https://archive.ics.uci.edu/ml/datasets/
    human+activity+recognition+using+smartphones>`_ Dataset.

    Args:
        root_dir (string): Root directory where dataset is downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root_dir>
                    └── UCIHARDataset
                        └── features.txt
                        ├── train
                        |   ├── InertialSignals
                        |   └── X_train.txt, y_train.txt, subject_train.txt
                        └── test
                            ├── InertialSignals
                            └── X_test.txt, y_test.txt, subject_test.txt
        what (string, optional): Can be 'features' or 'signals'. 'signals'
            corresponds to the raw signal data (9 types, 128 time steps),
            and 'features' corresponds to the processed data (561 features).
            The default is 'signals'.
        train (bool, optional): If True, creates dataset from train data,
            otherwise from test data.
        miss_rate (float, optional): The rate of the simulated missingness.
            The default is 0; meaning the complete data.
        scaler: An object with ``fit_transform`` and ``inverse_transform``
            methods. The default is ``sklearn.preprocessing.MinMaxScaler``.
        download (bool, optional): If If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
    """

    signal_types = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]

    labels = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]

    base_folder = "UCIHARDataset"
    data_url = ("https://archive.ics.uci.edu/ml/machine-learning"
                "-databases/00240/UCI%20HAR%20Dataset.zip")

    def __init__(
            self, root_dir: str, what: str = "signals",
            miss_rate: float = 0., train: bool = True,
            scaler: Any = MinMaxScaler(),
            download: bool = False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, self.base_folder)
        self.what = what
        self.miss_rate = miss_rate
        self.train = train
        self._split = "train" if self.train else "test"
        self.scaler = scaler

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You may use download=True to download it."
            )

        self.data, self.targets = self._load_data()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self.data[index], int(self.targets[index])

    @property
    def features_file(self) -> str:
        return os.path.join(self.data_dir, "features.txt")

    @property
    def data_file(self) -> str:
        return os.path.join(
            self.data_dir, self._split, f"X_{self._split}.txt"
        )

    @property
    def labels_file(self) -> str:
        return os.path.join(
            self.data_dir, self._split, f"y_{self._split}.txt"
        )

    @property
    def subjects_file(self) -> str:
        return os.path.join(
            self.data_dir, self._split, f"subject_{self._split}.txt"
        )

    @property
    def signal_files(self) -> List[str]:
        return [os.path.join(
            self.data_dir, self._split, "InertialSignals",
            file + f"_{self._split}.txt"
        ) for file in self.signal_types]

    def _load_data(self) -> Tuple[Tensor, Tensor]:
        """Load the data and targets, scale the data and introduce
        missing values.

        Returns:
            tuple: (data, activity_targets) where data is a tensor of shape
                :math:`(N, 128, 9)` if ``what='signals'``, and
                activity_targets is a tensor of shape :math:`(N, 1)`.
        """
        targets = (self._load_file(self.labels_file) - 1).reshape(-1)

        if self.what == 'signals':
            data = np.dstack(
                [self._load_file(file) for file in self.signal_files]
            )
        else:
            raise RuntimeError("features handling not implemented yet")

        # scale data
        n_features = data.shape[-1]
        data = self.scaler.fit_transform(data.reshape(-1, n_features)) \
            .reshape(data.shape)

        # introduce missing data
        data = ampute(data, self.miss_rate)

        return (
            Tensor(data).float(),
            Tensor(targets).long(),
        )

    @staticmethod
    def _load_file(filepath: str) -> np.ndarray:
        return pd.read_csv(filepath, delim_whitespace=True, header=None).values

    def _check_exists(self) -> bool:
        files = self.signal_files + [
            self.data_file, self.labels_file,
            self.subjects_file, self.features_file
        ]
        return all(os.path.isfile(file) for file in files)

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self.root_dir, exist_ok=True)

        try:
            print(f"Downloading UCI-HAR Dataset from {self.data_url}")
            download_and_unzip(
                self.data_url, extract_to=self.root_dir, remove_after=True
            )
        except URLError as error:
            raise RuntimeError(f"Failed to download UCI-HAR: \n{error}")


class OPPORTUNITY(Dataset):
    def __init__(self):
        super().__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        pass


if __name__ == '__main__':
    ucihar_dataset = UCIHAR('data', download=True)
