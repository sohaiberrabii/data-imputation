import random
from typing import Tuple, Callable
import numpy as np
import os
from torch import nn, Tensor
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from datasets import UCIHAR, OPPORTUNITY
from torch.utils.tensorboard import SummaryWriter
from models import TSImputer, GRUPlusFC
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer


def main(args):
    # Directory where results will be saved
    log_dir = os.path.join(args.log_dir, args.dataset)
    os.makedirs(log_dir, exist_ok=True)

    # Set logger
    writer_gan = SummaryWriter(
        os.path.join(log_dir, f"simple_gru_{args.miss_rate}")
    )

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    if args.dataset == 'UCIHAR':
        dataset = UCIHAR(
            root_dir=args.root_dir,
            missingness=args.missingness,
            miss_rate=args.miss_rate
        )
        dataset_train = dataset
        dataset_test = UCIHAR(root_dir=args.root_dir, train=False)
        n_classes = len(dataset.labels)
    elif args.dataset == 'OPPORTUNITY':
        dataset = OPPORTUNITY(
                root_dir=args.root_dir, seq_len=args.seq_len,
                overlap=args.overlap
        )
        dataset_train = OPPORTUNITY(
            root_dir=args.root_dir,
            split='train', overlap=args.overlap, seq_len=args.seq_len
        )
        dataset_test = OPPORTUNITY(
            root_dir=args.root_dir,
            split='test', seq_len=args.seq_len, overlap=args.overlap
        )
        n_classes = len(dataset.labels)
    else:
        raise NotImplementedError("BIRDS dataset not implemented")

    # Define UCI-HAR parameters
    _, n_timesteps, n_features = dataset.data.shape

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
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
    imputer.fit(dataloader, num_epochs=args.imputation_epochs)

    # Get the imputed datasets
    dataset_imputed = imputer.impute(x_miss=dataset_train.data)
    if args.dataset == 'OPPORTUNITY':
        dataset_test_imputed = imputer.impute(x_miss=dataset_test.data)

        # Create loader for imputed data
        dataloader_test = DataLoader(
            TensorDataset(dataset_test_imputed, dataset_test.targets),
            batch_size=args.batch_size,
            num_workers=args.workers
        )
    else:
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            num_workers=args.workers
        )

    # Create loader for imputed data
    dataloader_imp = DataLoader(
        TensorDataset(dataset_imputed, dataset_train.targets),
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


def classify(
        model, optimizer, train_loader,
        test_loader, num_epochs, metric,
        logger: SummaryWriter) -> None:
    for epoch in range(num_epochs):
        # Train
        train_loss, train_score = train(model, train_loader, optimizer, metric)

        # Test
        test_score = test(model, test_loader, metric)

        # Logging
        print(
            f"Epoch [{epoch}/{num_epochs}]\t"
            f" Loss: {train_loss:.6f}\t Train {metric}: {train_score:.4f}\t"
            f"Test {metric}: {test_score:.4f}"
        )
        logger.add_scalar("Train Loss", train_loss, epoch)
        logger.add_scalar(f"Train {metric}", train_score, epoch)
        logger.add_scalar(f"Test {metric}", test_score, epoch)


@torch.no_grad()
def mean_impute(x_miss: torch.Tensor) -> torch.Tensor:
    return torch.Tensor(SimpleImputer().fit_transform(
        x_miss.numpy().reshape(x_miss.shape[0], -1)
    )).reshape(x_miss.shape)


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(
        model: nn.Module,
        loader: DataLoader,
        optimizer, metric) -> Tuple[float, float]:
    # Init metrics
    losses = AverageMeter()
    scores = AverageMeter()

    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

        # record accuracy
        predicted = output.argmax(1)
        score = compute_score(y, predicted, metric)
        scores.update(score, n=y.size(0))

        # record loss
        losses.update(loss.item(), n=y.size(0))

        loss.backward()
        optimizer.step()

    return losses.avg, scores.avg


@torch.no_grad()
def test(model, loader, metric) -> float:
    # Init metrics
    scores = AverageMeter()

    for i, (x, y) in enumerate(loader):
        # Classify batch with model
        output = model(x)

        # Compute score
        predicted = output.argmax(1)
        score = compute_score(y, predicted, metric)
        scores.update(score, n=y.size(0))

    return scores.avg


@torch.no_grad()
def compute_score(y: Tensor, y_pred: Tensor, metric: Callable) -> float:
    if metric == 'Accuracy':
        return (y_pred == y).sum().item() / y.size(0)
    elif metric == 'Fscore':
        return f1_score(y, y_pred, average='weighted')


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
        choices=['UCIHAR', 'OPPORTUNITY', 'BIRDS'],
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
        '--seq_len',
        help='Length of subsequences of time series',
        default=200,
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
