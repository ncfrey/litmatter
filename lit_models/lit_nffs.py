import os.path as osp

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, ModuleList, BatchNorm1d, MSELoss

# from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from nff.data import Dataset, split_train_validation_test, collate_dicts, to_tensor
from nff.train import get_trainer, get_model, load_model, hooks, metrics, evaluate
from nff.train.loss import build_mse_loss


class LitNFF(LightningModule):
    def __init__(self, model_params, loss_params, lr=3e-4):
        super().__init__()

        model = get_model(model_params, model_type=model_params["model_type"])
        self.model = model
        self.save_hyperparameters()
        self.lr = lr

        self.loss_fn = build_mse_loss(loss_coef=loss_params)

    def training_step(self, batch, batch_idx: int):
        outputs = self.model(batch)
        loss = self.loss_fn(batch, outputs)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        torch.set_grad_enabled(True)  # needed for nffs
        outputs = self.model(batch)
        loss = self.loss_fn(batch, outputs)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int):
        torch.set_grad_enabled(True)  # needed for nffs
        outputs = self.model(batch)
        loss = self.loss_fn(batch, outputs)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def forward(self, x):
        torch.set_grad_enabled(True)  # needed for nffs
        return self.model(x)
