import os.path as osp

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, ModuleList, BatchNorm1d, MSELoss

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn import DimeNet, SchNet

from utils import MyTransform, Batch, Complete

target = 0
dim = 64


class LitDimeNet(LightningModule):
    def __init__(self, target=0):
        """Define PyTorch model."""
        super().__init__()
        model = DimeNet(
            hidden_channels=128,
            out_channels=1,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
        )
        self.model = model
        self.save_hyperparameters()
        self.target = target
        self.loss_fn = MSELoss()

    def training_step(self, batch, batch_idx: int):
        pred = self.model(batch.z, batch.pos, batch.batch)
        loss = self.loss_fn(pred.view(-1), batch.y[:, self.target])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        pred = self.model(batch.z, batch.pos, batch.batch)
        loss = self.loss_fn(pred.view(-1), batch.y[:, self.target])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int):
        pred = self.model(batch.z, batch.pos, batch.batch)
        loss = self.loss_fn(pred.view(-1), batch.y[:, self.target])
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=5, min_lr=0.00001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class LitSchNet(LightningModule):
    def __init__(self, target):
        super().__init__()
        self.save_hyperparameters()

        model = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
        )
        self.model = model
        self.target = target

        self.loss_fn = MSELoss()

    def training_step(self, batch: Batch, batch_idx: int):
        pred = self.model(batch.z, batch.pos, batch.batch)
        loss = self.loss_fn(pred.view(-1), batch.y[:, self.target])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        pred = self.model(batch.z, batch.pos, batch.batch)
        loss = self.loss_fn(pred.view(-1), batch.y[:, self.target])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: Batch, batch_idx: int):
        pred = self.model(batch.z, batch.pos, batch.batch)
        loss = self.loss_fn(pred.view(-1), batch.y[:, self.target])
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=5, min_lr=0.00001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class LitNNConv(LightningModule):
    def __init__(self, num_features, dim=64):
        super().__init__()
        self.save_hyperparameters()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr="mean")
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)

    def training_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=5, min_lr=0.00001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
