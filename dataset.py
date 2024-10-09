import torch
from pytorch_lightning import LightningModule, LightningDataModule

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration
import pandas as pd
from pandas import DataFrame
from typing import Mapping, Tuple, Dict
from torch import Tensor

class QAGModel(LightningModule):
    def __init__ (this, 
                  model: T5ForConditionalGeneration,
                  new_tokenizer_len: int,
                  optimizer,
                  optimizer_lr: float = 1e-4):
        super().__init__()
        this.model = model
        this.model.resize_token_embeddings(new_tokenizer_len)
        this.lr = optimizer_lr
        this.opt = optimizer

    def forward(this, input_ids, attention_mask, labels=None):
        output: Tensor = this.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
        return output.loss, output.logits

    def training_step(this, batch: Dict, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(this, batch: Dict, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(this, batch: Dict, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def exe_step(this, batch: Dict, batch_indx: int):
        input_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]
        labels: Tensor = batch["labels"]
        loss, output = this(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        return loss

    def configure_optimizers(this):
        return this.opt(this.parameters(),
                        lr=this.lr)


class QAGDataset(Dataset):
    def __init__(this,
                 data: pd.DataFrame,
                 tokenizer,
                 answer_token: str,
                 context_token: str,
                 question_token: str,
                 max_source_token_len: int = 512,
                 max_target_token_len: int = 512):
        this.tokenizer = tokenizer
        this.answer_token = answer_token
        this.context_token = context_token
        this.question_token = question_token
        this.max_source_token_len = max_source_token_len
        this.max_target_token_len = max_target_token_len
        this.data = data

    def __len__(this):
        return this.data.shape[0]

    def __getitem__(this, index: int):
        row = this.data.iloc[index]
        source_encoding = this.tokenizer(
            "{} {}".format(this.context_token,
                                    row["context"]),
            max_length=this.max_source_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        target_encoding = this.tokenizer(
            "{} {} {} {}".format(this.answer_token,
                                    row["answer"],
                                    this.question_token,
                                    row['question']),
            max_length=this.max_target_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        labels = target_encoding['input_ids']
        labels[labels == 0] = -100
        return dict(
            input_ids=source_encoding['input_ids'].flatten(),
            attention_mask=source_encoding['attention_mask'].flatten(),
            labels=labels.flatten())


class QAGDataModule(LightningDataModule):
    def __init__(this,
                 train_df,
                 val_df,
                 test_df,

                 tokenizer,
                 question_token: str,
                 context_token: str,
                 answer_token: str,

                 batch_size: int,
                 valid_batch_size: int,
                 max_source_token_len,
                 max_target_token_len,
                 ):
        super().__init__()
        this.train_df: DataFrame = DataFrame(train_df)
        this.val_df: DataFrame = DataFrame(val_df)
        this.test_df: DataFrame = DataFrame(test_df)

        this.tokenizer = tokenizer
        this.context_token = context_token
        this.question_token = question_token
        this.answer_token = answer_token
        this.batch_size = batch_size
        this.val_batch_size = valid_batch_size

        this.max_source_token_len = max_source_token_len
        this.max_target_token_len = max_target_token_len

    def setup(this, stage: str = None) -> None:
        this.train_dataset = QAGDataset(this.train_df, this.tokenizer,
                                       this.answer_token,
                                       this.context_token,
                                       this.question_token,
                                       this.max_source_token_len,
                                       this.max_target_token_len)

        this.val_dataset = QAGDataset(this.val_df, this.tokenizer,
                                     this.answer_token,
                                     this.context_token,
                                     this.question_token,
                                     this.max_source_token_len,
                                     this.max_target_token_len)

        this.test_dataset = QAGDataset(this.test_df, this.tokenizer,
                                      this.answer_token,
                                      this.context_token,
                                      this.question_token,
                                      this.max_source_token_len,
                                      this.max_target_token_len)

    def train_dataloader(this):
        return DataLoader(this.train_dataset, batch_size=this.batch_size,
                          shuffle=True, num_workers=2)

    def val_dataloader(this):
        return DataLoader(this.val_dataset, batch_size=this.val_batch_size,
                          num_workers=2)

    def test_dataloader(this):
        return DataLoader(this.test_dataset, batch_size=this.val_batch_size,
                          num_workers=2)