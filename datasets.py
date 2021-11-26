import os
from typing import List, Tuple, Any
from numpy import ndarray
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
            miss_rate: float = 0., missingness: str = "sequence",
            train: bool = True, scaler: Any = MinMaxScaler(),
            download: bool = False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, self.base_folder)
        self.what = what
        self.miss_rate = miss_rate
        self.missingness = missingness
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
            raise NotImplementedError("features handling not implemented")

        # scale data
        n_features = data.shape[-1]
        data = self.scaler.fit_transform(
            data.reshape(-1, n_features)
        ).reshape(data.shape)

        # introduce missing data
        data = ampute(data, self.miss_rate, missingness=self.missingness)

        return (
            Tensor(data).float(),
            Tensor(targets).long(),
        )

    # TODO: inverse transform of scaling
    def save_data(self, root_dir: str, data: ndarray) -> None:
        # Make directory for data files
        data_dir = os.path.join(
            root_dir, f"imputed_ucihar_{int(self.miss_rate * 100)}"
        )
        os.makedirs(data_dir, exist_ok=True)

        # Save data as separate signals
        for i, signal in enumerate(self.signal_types):
            np.savetxt(os.path.join(data_dir, f"{signal}.txt"), data[:, :, i])

    @staticmethod
    def _load_file(filepath: str) -> ndarray:
        return pd.read_csv(
            filepath, delim_whitespace=True, header=None
        ).to_numpy()

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
                self.data_url, extract_to=self.root_dir, remove_after=False
            )
        except URLError as error:
            raise RuntimeError(f"Failed to download UCI-HAR: \n{error}")


class OPPORTUNITY(Dataset):

    n_subjects = 4
    n_adl_runs = 5
    n_features = 242
    labels = ['Unknown', 'Stand', 'Walk', 'Sit', 'Lie']
    test_run_files = ["S2-ADL5.dat", "S3-ADL5.dat", "S4-ADL5.dat"]

    base_folder = "OpportunityUCIDataset"
    data_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "00226/OpportunityUCIDataset.zip")

    def __init__(
            self, root_dir: str,
            seq_len: int = 128,
            overlap: int = 0,
            what: str = 'adl',
            split: str = 'all',
            scaler: Any = MinMaxScaler(),
            download: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, self.base_folder)
        self.split = split
        self.what = what
        self.seq_len = seq_len
        self.overlap = overlap
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
    def adl_files(self) -> List[str]:
        adl_files = [
            f"S{subject}-ADL{run}.dat"
            for subject in range(1, self.n_subjects + 1)
            for run in range(1, self.n_adl_runs + 1)
        ]

        if self.split == 'train':
            adl_files = set(adl_files) - set(self.test_run_files)
        elif self.split == 'test':
            adl_files = self.test_run_files

        return [os.path.join(
            self.data_dir, "dataset", file
        ) for file in adl_files]

    @property
    def drill_files(self) -> List[str]:
        drill_run_files = [
            f"S{subject}-Drill.dat"
            for subject in range(1, self.n_subjects + 1)
        ]

        return [os.path.join(
            self.data_dir, "dataset", file
        ) for file in drill_run_files]

    def _check_exists(self) -> bool:
        files = self.drill_files + self.adl_files
        return all(os.path.isfile(file) for file in files)

    def _load_data(self) -> Tuple[Tensor, Tensor]:
        if self.what == 'adl':
            data = self._load_runs(self.adl_files)
        elif self.what == 'drill':
            data = self._load_runs(self.drill_files)
        elif self.what == 'all':
            data = self._load_runs(self.adl_files + self.drill_files)
        else:
            raise RuntimeError("what argument must be 'adl', 'drill' or 'all'")

        # scale sensor data (without time and labels)
        sensor_data = data[:, 1:self.n_features + 1]
        sensor_data = self.scaler.fit_transform(
            sensor_data.reshape(-1, self.n_features)
        ).reshape(-1, self.seq_len, self.n_features)

        # post-process targets of locomotion
        targets = data[:, self.n_features + 1]
        targets[targets == 4] = 3
        targets[targets == 5] = 4

        targets = targets.reshape(-1, self.seq_len)
        for i in range(targets.shape[0]):
            vals, count = np.unique(targets[i, :], return_counts=True)
            targets[i, 0] = vals[np.argmax(count)]
        targets = targets[:, 0]

        return (
            Tensor(sensor_data).float(),
            Tensor(targets).long(),
        )

    def _load_runs(self, files: List[str]) -> ndarray:
        return np.vstack([
                self._split_overlap(
                    data=self._load_file(file),
                    seq_len=self.seq_len, overlap=self.overlap
                )
                for file in files
        ])

    @staticmethod
    def _split_overlap(data: ndarray, seq_len: int, overlap: int) -> ndarray:
        q, mod = divmod(data.shape[0] - overlap, seq_len - overlap)

        step = seq_len - overlap
        overlapped_sequences = [
            data[i * step: i * step + seq_len, ...]
            for i in range(q)
        ]

        if mod:
            overlapped_sequences += [data[-seq_len:, ...]]

        return np.vstack(overlapped_sequences)

    @staticmethod
    def _load_file(filepath: str) -> ndarray:
        return pd.read_csv(
            filepath, delim_whitespace=True, header=None
        ).to_numpy()

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self.root_dir, exist_ok=True)

        try:
            print(f"Downloading Dataset from {self.data_url}")
            download_and_unzip(
                self.data_url, extract_to=self.root_dir, remove_after=False
            )
        except URLError as error:
            raise RuntimeError(f"Failed to download: \n{error}")


if __name__ == '__main__':
    # ucihar = UCIHAR(root_dir='data', download=True)
    opportunity = OPPORTUNITY(root_dir='data', what='drill')
    print(f"Number of samples: {len(opportunity)}")
