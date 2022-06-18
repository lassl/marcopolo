import math
import os

import torch
from torch.optim import AdamW
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from utils import AverageMeter, gather


class DPRTrainer:
    def __init__(
        self,
        model,
        datamodule,
        tokenizer,
        accelerator,
        logger,
        learning_rate,
        num_train_epochs,
        max_train_steps,
        num_warmup_steps,
        logging_steps,
        gradient_accumulation_steps,
    ):
        self.model = model
        self.datamodule = datamodule
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.logger = logger
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.logging_steps = logging_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_dataloader = datamodule.train_dataloader()
        self.valid_dataloader = datamodule.valid_dataloader()
        self.metric_acc = Accuracy()

    def setup(self):
        self.adjust_train_steps()
        self.configure_optimizers()
        self.acccelerate_prepare()
        self.metrics_to_device()
        if self.accelerator.is_main_process:
            os.makedirs("models", exist_ok=True)
            os.makedirs("models/q_encoder", exist_ok=True)
            os.makedirs("models/c_encoder", exist_ok=True)

    def adjust_train_steps(self):
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.gradient_accumulation_steps / self.accelerator.num_processes
        )
        self.max_train_steps = self.num_train_epochs * self.num_update_steps_per_epoch

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-4,
            },
            {
                "params": [p for n, p in self.model.q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in self.model.c_encoder.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-4,
            },
            {
                "params": [p for n, p in self.model.c_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate,)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.max_train_steps,
        )

    def acccelerate_prepare(self):
        (
            self.model.q_encoder,
            self.model.c_encoder,
            self.train_dataloader,
            self.valid_dataloader,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.model.q_encoder, 
            self.model.c_encoder, 
            self.train_dataloader, 
            self.valid_dataloader, 
            self.optimizer,
        )

    def metrics_to_device(self):
        device = self.accelerator.device
        self.metric_acc.to(device)

    def save_model(self):
        self.accelerator.wait_for_everyone()
        uw_q_encoder = self.accelerator.unwrap_model(self.model.q_encoder)
        uw_c_encoder = self.accelerator.unwrap_model(self.model.c_encoder)
        uw_q_encoder.save_pretrained("models/q_encoder", save_function=self.accelerator.save)
        uw_c_encoder.save_pretrained("models/c_encoder", save_function=self.accelerator.save)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained("models/q_encoder")
            self.tokenizer.save_pretrained("models/c_encoder")

    def fit(self):
        best_acc = 0
        for epoch in range(self.num_train_epochs):
            self.epoch = epoch
            self.train()
            acc = self.validate()

            if best_acc < acc:
                self.save_model()
                best_acc = acc

    def train(self):
        losses = AverageMeter()
        self.model.q_encoder.train()
        self.model.c_encoder.train()
        tepoch = tqdm(range(self.num_update_steps_per_epoch), unit="ba", disable=not self.accelerator.is_main_process,)
        for step, batch in enumerate(self.train_dataloader):
            tepoch.set_description(f"Epoch {self.epoch}")
            q_batch = {key[2:]: value for key, value in batch.items() if key.startswith("q_")}
            p_batch = {key[2:]: value for key, value in batch.items() if key.startswith("p_")}
            n_batch = {key[2:]: value for key, value in batch.items() if key.startswith("n_")}

            q_embedding = self.model.q_encoder(**q_batch).pooler_output
            p_embedding = self.model.c_encoder(**p_batch).pooler_output
            n_embedding = self.model.q_encoder(**n_batch).pooler_output

            q_embedding = gather(q_embedding, dim=0)
            p_embedding = gather(p_embedding, dim=0)
            n_embedding = gather(n_embedding, dim=0)

            c_embedding = torch.cat((p_embedding, n_embedding), dim=0)

            logits = torch.matmul(q_embedding, c_embedding.t())
            preds = logits.argmax(dim=-1)
            targets = torch.arange(0, q_embedding.size(0)).to(self.accelerator.device)
            loss = self.criterion(logits, targets)
            losses.update(loss.item(), targets.size(0))
            
            self.metric_acc.update(
                preds=preds, 
                target=targets,
            )
            loss = loss / self.gradient_accumulation_steps
            self.accelerator.backward(loss)

            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                tepoch.update(1)

            if (step + 1) % self.logging_steps == 0 or (step + 1) == len(self.train_dataloader):
                acc = self.metric_acc.compute()
                tepoch.set_postfix(
                    loss=f"{losses.avg:.4f}", acc=f"{(100.0 * acc):.2f}%",
                )
                losses.reset()
                self.metric_acc.reset()

    def validate(self):
        self.model.q_encoder.eval()
        self.model.c_encoder.eval()
        p_embedding, q_embedding = [], []
        with torch.no_grad():
            with tqdm(
                self.valid_dataloader, unit="ba", disable=not self.accelerator.is_main_process, desc="Validation",
            ) as tepoch:
                for _, batch in enumerate(tepoch):
                    q_batch = {key[2:]: value for key, value in batch.items() if key.startswith("q_")}
                    p_batch = {key[2:]: value for key, value in batch.items() if key.startswith("p_")}
                    q_embedding.append(self.model.q_encoder(**q_batch).pooler_output)
                    p_embedding.append(self.model.c_encoder(**p_batch).pooler_output)

            q_embedding = torch.cat(q_embedding, dim=0)
            p_embedding = torch.cat(p_embedding, dim=0)

            q_embedding = gather(q_embedding, dim=0)
            p_embedding = gather(p_embedding, dim=0)

            logits = torch.matmul(q_embedding, p_embedding.t())
            targets = torch.arange(0, q_embedding.size(0)).to(self.accelerator.device)
            loss = self.criterion(logits, targets)

        top1, top5, top20, top100 = accuracy(logits, targets, topk=(1, 5, 20, 100))

        self.logger.info(
            f"[Epoch {self.epoch}] top1={100*top1:.2f}%, top5={100*top5:.2f}%, top20={100*top20:.2f}%, top100={100*top100:.2f}%, val_loss={loss:.4f}"
        )
        return top1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res
