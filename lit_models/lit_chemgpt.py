import torch
import torch.nn.functional as F
from datasets import load_dataset
import logging
import math

import sys
import argparse

import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins import DeepSpeedPlugin

import numpy as np
import pandas as pd 

import copy
import os
import socket
import random
from time import time
import re

import selfies as sf

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import seaborn as sns
import matplotlib.pyplot as plt

import datasets
import transformers
from datasets import load_dataset, Dataset, DatasetDict, load_metric, load_from_disk
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer, Regex
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


class LitChemGPT(LightningModule):
    def __init__(self, model_size=16, dataset_size=2, num_epochs=2, lr=2e-5, from_pretrained=None, warmup_steps=100,
        reload_weight_path=None, reload_config_path=None, tokenizer_dir='.', logs_dir='loss_logs', cache_path='.'):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.seed = 42
        self.batch_size = 16
        self.tokenizer_dir = tokenizer_dir
        self.logs_dir = logs_dir
        self.cache_path = cache_path
        
        self.warmup_steps = warmup_steps
       # self.reload_weight_path = reload_weight_path
       # self.reload_config_path = reload_config_path
        
        # tokenizer
        tokenizer_dir = self.tokenizer_dir
        logs_dir = self.logs_dir
        num_proc = 16

        new_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir+'pubchem10M_tokenizer/')
        new_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
        # Instantiate the model
        # random weights initialization
        cache_path = self.cache_path
        num_layers = 24  # default 24
        hidden_size = model_size  # default 2048, must be divisible by num_heads (16)
        config = GPTNeoConfig(num_layers=num_layers, attention_types=[[['global', 'local'], num_layers // 2]],
                              hidden_size=hidden_size,
                              cache_path=cache_path)
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
        
        self.log('train_loss', loss, prog_bar=False, on_step=True, on_epoch=True)

        #self.logger.experiment.add_scalar("Train/step/loss", loss, self.trainer.global_step)
        #self.logger.experiment.add_scalar("Train/step/perplexity", metrics["perplexity"], self.trainer.global_step)
        
        return {'loss': loss,'progress_bar': {'loss': loss}, 'metrics': metrics}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_perplexity = math.exp(avg_loss)
        metrics = {"loss": avg_loss, "perplexity": avg_perplexity}

#         self.logger.experiment.add_scalar("Train/epoch/loss", avg_loss, self.current_epoch)
#         self.logger.experiment.add_scalar("Train/epoch/perplexity", avg_perplexity, self.current_epoch)


    def validation_step(self, batch, batch_idx):

        outputs = self.model(**batch)
        loss = outputs.loss
        metrics = {}
        
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return {'val_loss': loss, 'metrics': metrics}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.val_loss = avg_loss.item()
        if self.global_rank == 0:
            print('Val loss: {}'.format(self.val_loss))
        avg_perplexity = math.exp(avg_loss)
        if self.global_rank == 0:
            print('Val perplexity: {}'.format(avg_perplexity))

        metrics = {"val_loss": avg_loss, "val_perplexity": avg_perplexity}


    def sync_across_gpus(self, t):   # t is a tensor       
        # a work-around function to sync outputs across multiple gpus to compute a metric
        gather_t_tensor = [torch.ones_like(t) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_t_tensor, t)
        return torch.cat(gather_t_tensor)

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        if isinstance(outputs[0],tuple):
            loss, metrics = outputs[0]
        else:
            loss = outputs.loss
            metrics = {}
        return {'test_loss': loss, 'metrics': metrics, "test_perplexity": metrics["perplexity"]}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_loss = avg_loss.item()
        perplexities = torch.stack([x['test_perplexity'] for x in outputs])
        avg_perplexity = perplexities.mean()
        metrics = {"test_loss": avg_loss, "test_perplexity": avg_perplexity}
        print('avg perplexity:', avg_perplexity)

        if self.trainer.use_ddp:
            avg_perplexity_all = self.sync_across_gpus(perplexities).mean()
        print('average perplexity (all)', avg_perplexity_all)

        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=100)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler}}