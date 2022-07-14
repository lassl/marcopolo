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
        args,
        model,
        datamodule,
        tokenizer,
        accelerator,
        logger,
        # learning_rate,
        # num_train_epochs,
        # max_train_steps,
        # num_warmup_steps,
        # logging_steps,
        # gradient_accumulation_steps,
        # save_directory,
    ):
        # self.model = model
        self.model = model.to(accelerator.device)
        self.datamodule = datamodule
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.logger = logger
        self.learning_rate = args.learning_rate
        self.num_train_epochs = args.num_train_epochs
        self.max_train_steps = args.max_train_steps
        self.num_warmup_steps = args.num_warmup_steps
        self.logging_steps = args.logging_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.save_directory = args.save_directory
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
            os.makedirs("models/tokenizer", exist_ok=True)

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
                    p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-4,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate,)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.max_train_steps,
        )

    def acccelerate_prepare(self):
        (
            self.model,
            self.train_dataloader,
            self.valid_dataloader,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader, 
            self.valid_dataloader, 
            self.optimizer,
        )

    def metrics_to_device(self):
        device = self.accelerator.device
        self.metric_acc.to(device)

    def fit(self):
        best_acc = 0
        for epoch in range(self.num_train_epochs):
            self.epoch = epoch
            self.train()
            acc = self.validate()

            if best_acc < acc:
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                    self.save_directory, 
                    is_main_process=self.accelerator.is_main_process, 
                    save_function=self.accelerator.save,
                )
                if self.accelerator.is_main_process:
                    self.tokenizer.save_pretrained(os.path.join(self.save_directory, "tokenizer"))
                best_acc = acc


    def train(self):
        losses = AverageMeter()
        self.model.train()
        tepoch = tqdm(range(self.num_update_steps_per_epoch), unit="ba", disable=not self.accelerator.is_main_process,)
        for step, batch in enumerate(self.train_dataloader):
            tepoch.set_description(f"Epoch {self.epoch}")
            q_batch = {key[2:]: value for key, value in batch.items() if key.startswith("q_")}
            p_batch = {key[2:]: value for key, value in batch.items() if key.startswith("p_")}
            n_batch = {key[2:]: value for key, value in batch.items() if key.startswith("n_")}

            q_embedding = self.model(**q_batch, is_query=True).contiguous()
            p_embedding = self.model(**p_batch).contiguous()
            n_embedding = self.model(**n_batch).contiguous()

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
        self.model.eval()
        p_embedding, q_embedding = [], []
        with torch.no_grad():
            with tqdm(
                self.valid_dataloader, unit="ba", disable=not self.accelerator.is_main_process, desc="Validation",
            ) as tepoch:
                for _, batch in enumerate(tepoch):
                    q_batch = {key[2:]: value for key, value in batch.items() if key.startswith("q_")}
                    p_batch = {key[2:]: value for key, value in batch.items() if key.startswith("p_")}
                    q_embedding.append(self.model(**q_batch, is_query=True))
                    p_embedding.append(self.model(**p_batch))

            q_embedding = torch.cat(q_embedding, dim=0).contiguous()
            p_embedding = torch.cat(p_embedding, dim=0).contiguous()

            q_embedding = gather(q_embedding, dim=0)
            p_embedding = gather(p_embedding, dim=0)

            logits = torch.matmul(q_embedding, p_embedding.t())
            targets = torch.arange(0, q_embedding.size(0)).to(self.accelerator.device)
            loss = self.criterion(logits, targets)
        topk = accuracy(logits, targets, topk=(1, 5, 20, 100))
        mesg = f"[Epoch {self.epoch}" + "".join([f" top{k}={100*acc:.2f}%," for k, acc in topk]) + f" val_loss={loss:.4f}"
        self.logger.info(mesg)
        return topk[0][1]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    assert len(topk) > 0, "topk must be provided."
    with torch.no_grad():
        maxk = min(max(topk), len(output[-1]))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if len(correct) >= k:
                correct_k = correct[:k].reshape(-1).float().sum()
                res.append([k, correct_k.mul_(1.0 / batch_size)])
        return res