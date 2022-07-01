# import pkgs
import os

from typing import Union, List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from transformers import AutoTokenizer, TrainingArguments, AutoModel
from torch.utils.data import DataLoader
from accelerate import Accelerator

from datasets import load_dataset

from src.utils.seed import seed_config, seed_everything
from src.marcopolo.dpr.processor import DataCollatorForDPR, make_ex
from src.marcopolo.dpr.model import Biencoder
from src.marcopolo.dpr.trainer import Trainer
from utils.logger import get_logger
from accelerate import DistributedDataParallelKwargs

# get_params
@dataclass
class DataArguments:
    data_name_or_path: str = field(default=None, metadata={"help": "Data Source name or path"})
    preprocess_numproc: int = 1
    min_negative_size: int = field(default=9)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class DPRTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-5)
    lr_scheduler_type: str = field(default="linear")
    output_dir: str = field(default="/home/iron/project/marcopolo/models/")


if __name__ == "__main__":
    seed_everything(seed_config.seed)
    data_args = DataArguments()
    model_args = ModelArguments()
    training_args = DPRTrainingArguments()

    data_args.data_name_or_path = "Tevatron/msmarco-passage"
    data_args.preprocess_numproc = os.cpu_count() / 2
    model_args.model_name_or_path = "roberta-base"

    # https://github.com/huggingface/accelerate/issues/24
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    with accelerator.local_main_process_first():
        dataset = load_dataset(data_args.data_name_or_path)
        dataset = dataset.filter(lambda ex: len(ex["negative_passages"]) >= data_args.min_negative_size)
        dataset = dataset["train"].train_test_split(0.1)
        prep_data = dataset.map(lambda ex: make_ex(ex), num_proc=int(data_args.preprocess_numproc))

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    query_model = AutoModel.from_pretrained(model_args.model_name_or_path)
    pas_model = AutoModel.from_pretrained(model_args.model_name_or_path)

    data_collator = DataCollatorForDPR(tokenizer)
    model = Biencoder(query_model, pas_model)

    train_data_loader = DataLoader(
        prep_data["train"],
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        drop_last=True,
    )

    valid_data_loader = DataLoader(
        prep_data["test"],
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        drop_last=True,
    )

    training_args.num_train_epochs = 10

    if accelerator.is_main_process:
        logger = get_logger(name="train", file_path="./logging/train.log", stream=True)
    else:
        logger = None

    trainer = Trainer(
        model=model,
        accelerator=accelerator,
        train_dataloader=train_data_loader,
        valid_dataloader=valid_data_loader,
        args=training_args,
        logger=logger,
    )

    trainer.train()
