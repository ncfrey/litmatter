from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader

import datasets
import transformers
from datasets import load_dataset, Dataset, DatasetDict, load_metric, load_from_disk
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    Regex,
)
from transformers import BertTokenizerFast, PreTrainedTokenizerFast
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    set_seed,
    DataCollatorForLanguageModeling,
)


class ChemDataModule(LightningDataModule):
    def __init__(
        self, data_dir, tokenizer_dir, batch_size=16, num_workers=4, debug=False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_dir = tokenizer_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.new_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_dir)
        self.new_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.new_tokenizer, mlm=False
        )

    def prepare_data(self):
        self.lm_datasets = load_from_disk(self.data_dir)

    def setup(self, stage=None):
        self.train_set = self.lm_datasets["train"]
        self.val_set = self.lm_datasets["validation"]

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            pin_memory=True,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers
        )
