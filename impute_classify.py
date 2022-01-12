import random
from typing import Tuple, Callable
import numpy as np
import os
from torch import nn, Tensor
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from datasets import UCIHAR
from models import TSImputer, GRUPlusFC
from utils import classify


def main(args):
    # Directory where results will be saved
    log_dir = os.path.join(args.log_dir, args.dataset)
    os.makedirs(log_dir, exist_ok=True)

    # Set logger
    writer_gan = SummaryWriter(
        os.path.join(log_dir, f"simple_gru_{args.miss_rate}_{args.imputation_epochs}")
    )

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    ucihar_observed = UCIHAR(root_dir=args.root_dir)
    ucihar_train = UCIHAR(
        root_dir=args.root_dir,
        missingness=args.missingness,
        miss_rate=args.miss_rate,
        download=True
    )
    ucihar_test = UCIHAR(root_dir=args.root_dir, train=False)
    n_classes = len(dataset.labels)

    # Define UCI-HAR parameters
    _, n_timesteps, n_features = dataset.data.shape

    # Create the dataloader
    dataloader = DataLoader(
        ucihar_train,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    dataloader_test = DataLoader(
        ucihar_test,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    # Train the GAN imputer
    imputer = TSImputer(
        sequence_length=n_timesteps,
        input_size=n_features,
        hidden_size=args.hidden_size,
        alpha=args.alpha,
        lr=args.learning_rate
    )
    imputer.fit(
        dataloader,
        ucihar_observed.data,
        ucihar_train.data,
        num_epochs=args.imputation_epochs
    )

    # Get the imputed dataset
    ucihar_imputed = imputer.impute(x_miss=dataset_train.data)

    # Create loader for imputed data
    dataloader_imp = DataLoader(
        TensorDataset(ucihar_imputed, ucihar_train.targets),
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    # Init classifier
    clf = GRUPlusFC(
        n_timesteps, n_features,
        args.hidden_size, n_classes, last_only=True
    )

    # Optimizer
    optimizer = torch.optim.Adam(clf.parameters(), lr=args.learning_rate)

    # Train and evaluate the classifier
    classify(
        model=clf,
        optimizer=optimizer,
        train_loader=dataloader_imp,
        test_loader=dataloader_test,
        num_epochs=args.classification_epochs,
        metric=args.metric,
        logger=writer_gan
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-rd', '--root_dir',
        help='Root directory where the dataset is stored',
        default='data',
        type=str
    )
    parser.add_argument(
        '-d', '--dataset',
        help='Name of dataset',
        choices=['UCIHAR'],
        default='UCIHAR',
        type=str,
    )
    parser.add_argument(
        '-m', '--miss_rate',
        help='Introduced missing proportion in full datasets',
        default=0.2,
        type=float,
    )
    parser.add_argument(
        '--missingness',
        help='Missingness mechanism',
        default='sequence',
        type=str
    )
    parser.add_argument(
        '-bs', '--batch_size',
        help='Number of samples in mini-batch',
        default=128,
        type=int
    )
    parser.add_argument(
        '-a', '--alpha',
        help='Hyperparameter for training the generator',
        default=100,
        type=float
    )
    parser.add_argument(
        '-hz', '--hidden_size',
        help='Hidden size of the GRU used for imputation and classification',
        default=64,
        type=int
    )
    parser.add_argument(
        '--imputation_epochs',
        help='Number of training epochs for imputation',
        default=20,
        type=int
    )
    parser.add_argument(
        '--classification_epochs',
        help='Number of training epochs for classifier',
        default=100,
        type=int
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        help='Learning rate used for training the imputer',
        default=0.005,
        type=float
    )
    parser.add_argument(
        '-w', '--workers',
        help='Number of pytorch workers',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--log_dir',
        help='Directory where results are saved',
        default='runs',
        type=str,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Seed for random number generator',
        default=999,
        type=int,
    )
    parser.add_argument(
        '--metric',
        help='Metric used to evaluate the classification task',
        choices=['Accuracy', 'Fscore'],
        default='Accuracy',
        type=str,
    )
    args = parser.parse_args()

    main(args)
