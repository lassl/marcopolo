import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR

class Encoder_Trainer(LightningModule):
    def __init__(self, question_model, ctx_model, lr):
        super().__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric_acc = Accuracy()
        self.lr = lr    

    def emb_extract(self, sub_model, ids, attn_mask):
        sequence_output, pooled_output, hidden_states = sub_model(ids, attn_mask)
        return pooled_output

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.2
        )
        # Source: https://github.com/openai/CLIP/issues/107
        num_training_steps = len(self.trainer.datamodule.train_dataloader()) # single-gpu case
        scheduler = OneCycleLR(
            optimizer = optimizer,
            max_lr=self.lr,
            steps_per_epoch = num_training_steps,
            epochs = 200,
            pct_start = 0.01, # 2 epoch = 8k per 200
        )
        scheduler = {"scheduler": scheduler, "interval" : "step" }
        return [optimizer], [scheduler]

    def shared_step(self, batch):
        query, pos, neg = batch
        q_embedding = self.emb_extract(self.question_model, query['input_ids'], query['attention_mask'])
        p_embedding = self.ctx_model(self.ctx_model, pos['input_ids'], pos['attention_mask'])
        n_embedding = self.ctx_model(self.ctx_model, neg['input_ids'], neg['attention_mask'])

        c_embedding = torch.cat((p_embedding, n_embedding), dim=0)

        logits = torch.matmul(q_embedding, c_embedding.t())
        preds = logits.argmax(dim=-1)
        targets = torch.arange(0, q_embedding.size(0))
        loss = self.criterion(logits, targets)
        self.metric_acc.update(
                preds=preds, 
                target=targets,
            )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log_dict(
            {
                "train_loss": loss,
            },
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def training_step_end(self, step_output):
        return step_output

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        return {
                "val_loss": loss,
                "acc": self.metric_acc
            }

    def validation_step_end(self, step_output):
        return step_output

    def validation_epoch_end(self, outputs):        
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
            {
                "val_loss": val_loss
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
    def test_step(self, batch, batch_idx):
        return None

    def test_step_end(self, step_output):
        return step_output

    def test_epoch_end(self, outputs):
        return None