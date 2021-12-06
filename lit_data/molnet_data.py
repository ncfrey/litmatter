from typing import Optional

from pytorch_lightning import LightningDataModule

from torch_geometric.data import DataLoader as PyGDataLoader

from torch.utils.data import DataLoader

import deepchem as dc


class LitMolNet(LightningDataModule):
    def __init__(self, loader=dc.molnet.load_tox21, batch_size=16, num_workers=4):
        super().__init__()
        self.loader = loader
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """Download data if needed."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Apply transformations and split datasets."""
        task, df, trans = self.loader()
        train, valid, test = df
        train, valid, test = (
            train.make_pytorch_dataset(),
            valid.make_pytorch_dataset(),
            test.make_pytorch_dataset(),
        )

        self.train_dataset, self.val_dataset, self.test_dataset = train, valid, test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
