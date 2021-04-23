import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.val_check_interval = len(self.train_dataloader())
        self.current_val_loss = torch.tensor(float('inf'), device=(
            'cuda' if self.on_gpu else 'cpu'))

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        self.current_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': self.current_val_loss}
        print(self.current_val_loss.data)
        return {'val_loss': self.current_val_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.02)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='max',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )

        return [self.opt], [self.reduce_lr_on_plateau]


    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=512)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=512)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=512)

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i,
                       second_order_closure=None):
        self.opt.step()
        self.opt.zero_grad()
        if self.trainer.global_step % self.val_check_interval == 0:
            self.reduce_lr_on_plateau.step(self.current_val_loss)

mnist_model = MNISTModel()

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=0, auto_lr_find=True, show_progress_bar=True)    # min_epochs=1, max_epochs=1000
trainer.fit(mnist_model)

# https://pytorch-lightning.readthedocs.io/en/latest/advanced/transfer_learning.html
# https://pytorch-lightning.readthedocs.io/en/latest/ LESEN !!