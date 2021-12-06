import os.path as osp

from typing import Optional, List, NamedTuple

import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Sequential, Linear, ReLU, GRU, ModuleList, BatchNorm1d, MSELoss

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

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
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    set_seed,
    DataCollatorForLanguageModeling,
)

from transformers import GPTNeoForCausalLM, GPTNeoConfig


class LitHF(LightningModule):
    def __init__(
        self,
        tokenizer_dir,
        model_dir,
        from_pretrained=None,
        warmup_steps=100,
        reload_weight_path=None,
        reload_config_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = 2e-5
        self.seed = 42
        self.batch_size = 16
        self.model_dir = model_dir

        self.warmup_steps = warmup_steps

        # tokenizer
        self.tokenizer_dir = tokenizer_dir
        num_proc = 16

        new_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_dir)
        new_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Instantiate the model
        # random weights initialization
        num_layers = 8  # default 24
        hidden_size = 256  # default 2048, must be divisible by num_heads (16)
        config = GPTNeoConfig(
            num_layers=num_layers,
            attention_types=[[["global", "local"], num_layers // 2]],
            hidden_size=hidden_size,
            cache_path=self.model_dir,
        )
        config.vocab_size = new_tokenizer.vocab_size
        max_len_tokenized = 512
        config.max_length = max_len_tokenized

        # from config does NOT load weights! `from_pretrained` does
        model = GPTNeoForCausalLM(config)
        model.resize_token_embeddings(len(new_tokenizer))

        self.model = model

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        metrics = {}

        return {"loss": loss, "progress_bar": {"loss": loss}, "metrics": metrics}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_perplexity = math.exp(avg_loss)
        metrics = {"loss": avg_loss, "perplexity": avg_perplexity}

    def validation_step(self, batch, batch_idx):

        outputs = self.model(**batch)
        loss = outputs.loss
        metrics = {}

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return {"val_loss": loss, "metrics": metrics}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.val_loss = avg_loss.item()
        if self.global_rank == 0:
            print("Val loss: {}".format(self.val_loss))
        avg_perplexity = math.exp(avg_loss)
        if self.global_rank == 0:
            print("Val perplexity: {}".format(avg_perplexity))

        metrics = {"val_loss": avg_loss, "val_perplexity": avg_perplexity}

    def sync_across_gpus(self, t):  # t is a tensor
        # a work-around function to sync outputs across multiple gpus to compute a metric
        gather_t_tensor = [torch.ones_like(t) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_t_tensor, t)
        return torch.cat(gather_t_tensor)

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        if isinstance(outputs[0], tuple):
            loss, metrics = outputs[0]
        else:
            loss = outputs.loss
            metrics = {}
        return {
            "test_loss": loss,
            "metrics": metrics,
            "test_perplexity": metrics["perplexity"],
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.test_loss = avg_loss.item()
        perplexities = torch.stack([x["test_perplexity"] for x in outputs])
        avg_perplexity = perplexities.mean()
        metrics = {"test_loss": avg_loss, "test_perplexity": avg_perplexity}
        print("avg perplexity:", avg_perplexity)

        if self.trainer.use_ddp:
            avg_perplexity_all = self.sync_across_gpus(perplexities).mean()
        print("average perplexity (all)", avg_perplexity_all)

        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=100)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
