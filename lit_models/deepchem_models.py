import os.path as osp

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, ModuleList, BatchNorm1d, MSELoss

import deepchem as dc

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)


class LitDeepChem(LightningModule):
    def __init__(self, torch_model, lr=1e-2):
        """Define DeepChem TorchModel."""
        super().__init__()

        self.model = torch_model.model  # torch.nn.Module
        self.save_hyperparameters()
        self.lr = lr
        self.loss_fn = torch_model.loss

    def training_step(self, batch, batch_idx: int):
        # Modify for MolNet dataset as needed
        inputs = batch[0].float()
        y = batch[2].float()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        inputs = batch[0].float()
        y = batch[2].float()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int):
        inputs = batch[0].float()
        y = batch[2].float()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
