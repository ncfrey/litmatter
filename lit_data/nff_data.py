from typing import Optional

from pytorch_lightning import LightningDataModule


from torch.utils.data import DataLoader

from nff.data import Dataset, split_train_validation_test, collate_dicts, to_tensor
from nff.train import get_trainer, get_model, load_model, hooks, metrics, evaluate


class NFFDataModule(LightningDataModule):
    def __init__(self, path, batch_size=16, num_workers=4):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """Download data if needed."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Apply transformations and split datasets."""
        dataset = Dataset.from_file(self.path)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = split_train_validation_test(dataset, val_size=0.2, test_size=0.2)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_dicts,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_dicts,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_dicts,
        )
