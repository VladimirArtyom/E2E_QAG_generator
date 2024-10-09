from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer)
from pandas import DataFrame, concat
from torch import Tensor,load
from argparse import ArgumentParser, Namespace
from dataset import QAGDataModule, QAGDataset, QAGModel
from typing import List, Dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import datasets

def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="Voslannack/race_id")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--valid_batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)

    ### Model identifier
    parser.add_argument("--question_token", type=str, default="<question>")
    parser.add_argument("--answer_token", type=str, default="<answer>")
    parser.add_argument("--context_token", type=str, default="<context>")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/Thesis/qag_data/model_1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input_max_length", type=int, default=512)
    parser.add_argument("--target_max_length", type=int, default=512)
    parser.add_argument("--logs_dir", type=str, default="/content/drive/MyDrive/Thesis/qag_data/model_1/logs")
    parser.add_argument("--acc", type=str, default="gpu")
    parser.add_argument("--load_model", type=str, default="/content/drive/MyDrive/Thesis/qag_data/model_1/best_checkpoint.ckpt")

    return parser.parse_args()

if __name__ == "__main__":
    args: Namespace = parse_argument()
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": [args.question_token, args.answer_token,
                                      args.context_token, args.distractor_token]
    })
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device)
    data = datasets.load_dataset(args.dataset_path)
    train_data = data["train"]
    valid_data = data["validation"]
    test_data = data["test"]
    #driv.prepare_distractor_generator_model(model, len(tokenizer), AdamW(), args.lr)
    #driv.prepare_distractor_generator_datasets(DataFrame(train_data), DataFrame(valid_data), DataFrame(test_data))
    model_callbacks = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="best-checkpoint",
        monitor="val_loss",
        save_last=False,
        save_top_k=1,
        mode="min",
    )
    logger = TensorBoardLogger(args.logs_dir, name="qag_log")
    data_module = QAGDataModule(
        train_df=train_data,
        val_df=valid_data,
        test_df=test_data,
        tokenizer=tokenizer,
        question_token=args.question_token,
        context_token=args.context_token,
        answer_token=args.answer_token,
        batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        max_source_token_len=args.input_max_length,
        max_target_token_len=args.target_max_length
        )
    model_module = QAGModel(model,
                            len(tokenizer),
                            AdamW,
                            args.lr)
    trainer = Trainer(
        accelerator=args.acc,
        callbacks=model_callbacks,
        logger=logger,
        max_epochs=args.epochs
    )

    trainer.fit(model_module, 
                data_module)