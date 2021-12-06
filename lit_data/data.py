from typing import Optional

from pytorch_lightning import LightningDataModule

from torch_geometric.data import DataLoader as PyGDataLoader

from torch.utils.data import DataLoader

import deepchem as dc


class LitQM9(LightningDataModule):
    def __init__(self, datasets, batch_size=16, num_workers=4):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """Download data if needed."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Apply transformations and split datasets."""
        self.train_dataset, self.val_dataset, self.test_dataset = self.datasets

    def train_dataloader(self):
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
