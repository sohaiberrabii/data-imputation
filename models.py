from typing import Tuple
import torch
from torch.optim import Adam
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class TSImputer:
    """GAN-based Time series imputer.

    """
    def __init__(
            self,
            sequence_length,
            input_size,
            hidden_size,
            alpha: int = 100,
            lr: float = 0.005) -> None:
        self._generator = GRUPlusFC(
            sequence_length,
            input_size,
            hidden_size,
            output_size=input_size
        )
        self._discriminator = GRUPlusFC(
            sequence_length,
            input_size,
            hidden_size,
            output_size=input_size
        )

        self._optimizerG = Adam(self._generator.parameters(), lr=lr)
        self._optimizerD = Adam(self._discriminator.parameters(), lr=lr)

        # Hyperparameter
        self.alpha = alpha

        # Statistics
        self.lossesG = []
        self.lossesD = []

    def fit(
            self,
            dataloader: DataLoader,
            num_epochs: int = 100):
        for epoch in range(num_epochs):
            for i, (x, _) in enumerate(dataloader):
                # Sample random noise
                z = torch.rand(x.shape) * 0.01

                # Get binary mask from data, mask = 1 if observed
                mask = ~torch.isnan(x)

                # Combine random noise with observed data
                x = mask * torch.nan_to_num(x) + ~mask * z

                ###################################
                # (1) Update D: minimize BCE(M, D(G(x)))
                ###################################
                self._discriminator.zero_grad()

                # Generate imputed data with G
                x_g = self._generator(x)

                # Classify imputed data with D
                x_imp = mask * x + ~mask * x_g
                prob_d = self._discriminator(x_imp.detach())

                # Compute loss for D
                loss_d = nn.BCELoss()(prob_d, mask.float())
                # Compute gradients for D
                loss_d.backward()
                # Update D
                self._optimizerD.step()

                ###################################
                # (2) Update generator: minimize
                # MSE(M * x, M * G(x)) - (1 - M) * log(D(G(x))
                ###################################
                self._generator.zero_grad()

                # Classify imputed data once more since D got updated
                prob_d = self._discriminator(x_imp)

                # Compute loss for G
                loss_rec, loss_adv = self._generator_loss(x, x_g, mask, prob_d)
                loss_g = self.alpha * loss_rec - loss_adv
                # Compute gradients for G
                loss_g.backward()
                # Update G
                self._optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print(
                        f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t"
                        f"LossD: {loss_d.item():.6f}\t"
                        f"LossG_rec: {loss_rec.item():.6f}\t"
                        f"LossG_adv: {-loss_adv.item():.6f}"
                    )

                # Save Losses for plotting later
                self.lossesG.append(loss_g.item())
                self.lossesD.append(loss_d.item())

    @staticmethod
    def _generator_loss(
            x: Tensor, x_hat: Tensor,
            mask: Tensor, d_x: Tensor) -> Tuple[Tensor, Tensor]:
        return (
            nn.MSELoss()(mask * x, mask * x_hat) / torch.mean(mask.float()),
            torch.mean(~mask * torch.log(torch.add(d_x, 1e-8)))
        )

    @torch.no_grad()
    def impute(self, x_miss: Tensor) -> Tensor:

        z = torch.rand(x_miss.shape) * 0.01
        mask = ~torch.isnan(x_miss)
        x_noised = mask * torch.nan_to_num(x_miss) + ~mask * z

        x_g = self._generator(x_noised)

        x_imputed = mask * x_noised + ~mask * x_g

        return x_imputed


class GRUPlusFC(nn.Module):
    def __init__(
            self, sequence_length, input_size,
            hidden_size, output_size, last_only=False):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.last_only = last_only
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> Tensor:
        output, _ = self.gru(x, self._init_hidden(x.size(0)))
        output = self.fc(nn.ReLU(True)(output))
        if self.last_only:
            return output[:, -1]
        else:
            return nn.Sigmoid()(output)

    def _init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
