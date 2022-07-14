import logging
import warnings

import datasets
# import hydra
from accelerate import Accelerator, DistributedDataParallelKwargs
# from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, set_seed, TrainingArguments
from datasets import load_dataset
from utils import set_logging_default
from pathlib import Path
from argparse import ArgumentParser

from trainer import DPRTrainer
from model import DualEncoder
from datamodule import DPRDataModule

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_main_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    return args


def get_dataset():
    train_path = "./data/train/"
    corpora_dir = Path(train_path).absolute()
    list_of_file_paths = [str(file_path) for file_path in corpora_dir.rglob("*.json")]
    return load_dataset("json", data_files=list_of_file_paths, split="train")


# @hydra.main(config_path="configs/", config_name="dpr")
def main():

    args = get_main_args()
    nested_args = OmegaConf.load(args.config_path)
    model_args = nested_args.model
    data_args = nested_args.data
    collator_args = nested_args.collator
    training_args = TrainingArguments(**nested_args.training)


    # if config.get("seed"):
    set_seed(training_args.seed)

    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])

    set_logging_default(logger, accelerator)

    logger.info(f"Instantiate dataset")
    # dataset = datasets.load_dataset("Tevatron/msmarco-passage")
    dataset = get_dataset()
    # if config.debug:
    #     for split in dataset.keys():
    #         dataset[split] = dataset[split].select(range(200))

    logger.info(f"Instantiate tokenizer <{model_args.pretrained_name}>")
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_name)

    logger.info(f"Instantiate model <{model_args.pretrained_name}>")
    model = DualEncoder(model_args.pretrained_name)

    logger.info(f"Instantiate datamodule")
    # datamodule = instantiate(config.datamodule, dataset=dataset, tokenizer=tokenizer, accelerator=accelerator)
    datamodule = DPRDataModule(
        dataset=dataset, tokenizer=tokenizer, accelerator=accelerator, max_seq_length=256, 
        test_size=data_args.test_size, num_process=36, num_workers=36, args=training_args
        )

    logger.info(f"Instantiate trainer")
    # trainer = instantiate(
    #     config.trainer,
    #     model=model,
    #     datamodule=datamodule,
    #     tokenizer=tokenizer,
    #     accelerator=accelerator,
    #     logger=logger,
    # )
    trainer = DPRTrainer(
        args=training_args,
        model=model,
        datamodule=datamodule,
        tokenizer=tokenizer,
        accelerator=accelerator,
        logger=logger,
    )
    trainer.setup()
    trainer.fit()


if __name__ == "__main__":
    main()