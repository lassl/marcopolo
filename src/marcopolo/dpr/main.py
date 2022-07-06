import logging
import warnings

import datasets
import hydra
from accelerate import Accelerator, DistributedDataParallelKwargs
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import AutoTokenizer, set_seed
from utils import set_logging_default

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="dpr")
def main(config: DictConfig):
    if config.get("seed"):
        set_seed(config.seed)

    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])

    set_logging_default(logger, accelerator)

    logger.info(f"Instantiate dataset")
    dataset = datasets.load_dataset("Tevatron/msmarco-passage")
    if config.debug:
        for split in dataset.keys():
            dataset[split] = dataset[split].select(range(200))

    logger.info(f"Instantiate tokenizer <{config.model_name_or_path}>")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    logger.info(f"Instantiate model <{config.model_name_or_path}>")
    model = instantiate(config.model)

    logger.info(f"Instantiate datamodule")
    datamodule = instantiate(config.datamodule, dataset=dataset, tokenizer=tokenizer, accelerator=accelerator)

    logger.info(f"Instantiate trainer")
    trainer = instantiate(
        config.trainer,
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
