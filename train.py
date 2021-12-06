import argparse
import json
import numpy as np
import os

from torch_geometric.datasets import QM9
from torch_geometric.nn import DimeNet

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from .lit_models.models import LitDimeNet
from .lit_data.data import LitQM9


def format_args(config):
    if "TASK" in config:
        config["TASK"] = str(config["TASK"])
    if "BATCH_SIZE" in config:
        config["BATCH_SIZE"] = int(config["BATCH_SIZE"])
    if "NUM_EPOCHS" in config:
        config["NUM_EPOCHS"] = int(config["NUM_EPOCHS"])
    if "NUM_GPUS" in config:
        config["NUM_GPUS"] = int(config["NUM_GPUS"])


def train_from_config(config):
    task = config.task
    batch_size = config.batch_size
    num_train_epochs = config.num_epochs
    num_nodes = config.num_nodes

    seed_everything(42)

    dataset = QM9('data/QM9')
    target = 0
    _, datasets = DimeNet.from_qm9_pretrained('data/QM9', dataset, target)
    datamodule = LitQM9(datasets)
    datamodule.setup()
    model = LitDimeNet(target)

    # set up checkpointing
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1)

    trainer = Trainer(
        gpus=-1,  # number of GPUs per node
        num_nodes=num_nodes,
        accelerator='ddp',
        max_epochs=num_train_epochs,
        callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule=datamodule)
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--config",
                        type=str,
                        help="JSON Config filename for training parameters")
    args, unknown = parser.parse_known_args()
    train_from_config(args)
