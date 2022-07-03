import json
import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from pathlib import Path
import torch
import torchaudio
from datasets import load_dataset

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
# backbones
from transformers import AutoModel, AutoTokenizer, set_seed
# modules
from marcopolo.dpr.models.biencoder import Encoder_Trainer
from marcopolo.dpr.datasets.loader import DataPipeline

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger 

def get_checkpoint_callback(save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last= False,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback

def get_early_stop_callback() -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min"
    )
    return early_stop_callback

def save_hparams(args, save_path):
    save_config = OmegaConf.create(vars(args))
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(config=save_config, f= Path(save_path, "hparams.yaml"))

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/test"
    save_hparams(args, save_path)
    
    # logging
    import wandb
    wandb.init(config=args)
    wandb.run.name = f"exp/test"
    args = wandb.config

    question_encoder = AutoModel.from_pretrained(args.backbone)
    ctx_encoder = AutoModel.from_pretrained(args.backbone)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    msmarco = load_dataset('Tevatron/msmarco-passage')
    tf_dataset = msmarco["train"]

    runner = Encoder_Trainer(
        question_model = question_encoder, 
        ctx_model = ctx_encoder, 
        lr = args.lr   
    )
    pipeline = DataPipeline(
        hf_datasets = tf_dataset, 
        tokenizer = tokenizer, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers
    )

    logger = get_wandb_logger(runner)
    checkpoint_callback = get_checkpoint_callback(save_path)
    early_stop_callback = get_early_stop_callback()
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
                    max_epochs= args.max_epochs,
                    num_nodes= args.num_nodes,
                    accelerator='gpu',
                    strategy = args.strategy,
                    devices= args.gpus,
                    # strategy = DDPPlugin(find_unused_parameters=True),
                    logger=logger,
                    # log_every_n_steps=1,
                    sync_batchnorm=True,
                    resume_from_checkpoint=None,
                    replace_sampler_ddp=False,
                    callbacks=[
                        # early_stop_callback,
                        checkpoint_callback,
                        lr_monitor_callback
                    ],
                )

    trainer.fit(runner, datamodule=pipeline)

if __name__ == "__main__":
    # pipeline
    parser = ArgumentParser() 
    # runner
    parser.add_argument("--framework", default="dpr", type=str)
    parser.add_argument("--backbone", default="roberta-base", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--strategy", default="dp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=False, type=str2bool)
    args = parser.parse_args()
    main(args)