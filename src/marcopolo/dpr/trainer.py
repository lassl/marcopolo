from inspect import unwrap
import torch
from torch import nn
from typing import Union
from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from utils.dist import gather_step_tensor, get_idx_rank, MrrMetric
import math


class Trainer:
    """Trainer

    Attributes:
        model (`model`)
        device (str)
        loss_fn (Callable)
        optimizer (`optimizer`)
        scheduler (`scheduler`)
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        accelerator=None,
        train_dataloader=None,
        valid_dataloader=None,
        args: TrainingArguments = None,
        logger=None,
    ):
        self.model = model
        self.accelerator = accelerator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.args = args
        self.logger = logger
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = MrrMetric()
        self.best_mrr = None

    def train(self):

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader.dataset) / self.args.gradient_accumulation_steps / self.args.per_device_train_batch_size
        )
        self.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
        self.num_warmup_steps = math.ceil(self.max_train_steps * self.args.warmup_ratio)
        local_step = int(self.max_train_steps / self.accelerator.num_processes)
        scaled_train_step = int(self.num_warmup_steps / self.accelerator.num_processes)
        total_train_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes

        # Train object to Accelerator
        self.set_optimizer()
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.valid_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.valid_dataloader, self.scheduler
        )

        # Train!
        if self.accelerator.is_main_process:
            self.logger.info("***** Running training *****")
            self.logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
            self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
            self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            self.logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
            )
            self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            self.logger.info(f"  Num Warmup steps = {scaled_train_step}")
            self.logger.info(f"  Optimizer steps = {local_step}")

        for epoch in range(self.args.num_train_epochs):
            self._inner_train_loop(epoch)

        # (TO-DO) early_stop_check
        # (TO-DO) save performance of model

    def set_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 1e-4,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
        )
        # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )

    def _inner_train_loop(self, epoch_idx):
        self.model.train()
        self.step_loss_per_device = 0
        self.metric.clear()

        # define epoch from training args
        with tqdm(
            self.train_dataloader,
            desc=f"Train: {epoch_idx + 1}epochs ",
            disable=not self.accelerator.is_main_process,
        ) as train_epoch:
            for step, batch in enumerate(train_epoch):
                q_pooled_output, ctx_pooled_output = self.model(batch)

                c_all_tensor = self.accelerator.gather(ctx_pooled_output)
                scores = torch.matmul(q_pooled_output, c_all_tensor.t())
                labels = torch.arange(
                    self.accelerator.process_index * ctx_pooled_output.size(0),
                    self.accelerator.process_index * ctx_pooled_output.size(0) + q_pooled_output.size(0),
                ).to(
                    self.model.device
                )  # to.device

                loss = self.loss_fn(scores, labels)
                step_loss = [device_loss.item() for device_loss in gather_step_tensor(loss)]
                self.step_loss_per_device += sum(step_loss) / self.accelerator.num_processes

                pred_idx = get_idx_rank(scores, labels)
                step_pred = torch.cat(gather_step_tensor(pred_idx), dim=0)
                self.metric.update(step_pred.tolist())

                loss = loss / self.args.gradient_accumulation_steps
                self.accelerator.backward(loss)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    (step + 1) == len(self.train_dataloader)
                ):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if (step + 1) % 10 == 0:
                    avg_loss = self.step_loss_per_device / (step + 1)
                    train_epoch.set_postfix(
                        {
                            "step": step + 1,
                            "lr": "{:e}".format(self.scheduler.get_last_lr()[0]),
                            "loss": "{:.2f}".format(avg_loss),
                        }
                    )

            self.train_mean_loss = self.step_loss_per_device / len(self.train_dataloader)
            mrr_score = self.metric.compute()
            msg = f"Epoch {epoch_idx + 1}, loss: {self.train_mean_loss}, learning_rate: {self.scheduler.get_last_lr()[0]:e}, mrr: {mrr_score}"

            if self.accelerator.is_main_process:
                self.logger.info(msg) if self.logger else print(msg)

        if self.valid_dataloader:
            self._inner_valid_loop(epoch_idx)

            if self.best_mrr is None or self.valid_mrr > self.best_mrr:
                self.save_model()
                self.best_mrr = self.valid_mrr
        else:
            self.save_model()

    def _inner_valid_loop(self, epoch_idx):
        self.model.eval()
        self.valid_step_loss = 0
        self.metric.clear()

        # define epoch from training args
        with torch.no_grad():
            with tqdm(
                self.valid_dataloader,
                desc=f"Valid: {epoch_idx + 1}epochs ",
                disable=not self.accelerator.is_main_process,
            ) as valid_epoch:
                for step, batch in enumerate(valid_epoch):
                    q_pooled_output, ctx_pooled_output = self.model(batch)

                    c_all_tensor = self.accelerator.gather(ctx_pooled_output.contiguous())
                    scores = torch.matmul(q_pooled_output, c_all_tensor.t())
                    labels = torch.arange(
                        self.accelerator.process_index * ctx_pooled_output.size(0),
                        self.accelerator.process_index * ctx_pooled_output.size(0) + q_pooled_output.size(0),
                    ).to(
                        self.model.device
                    )  # to.device

                    loss = self.loss_fn(scores, labels)
                    step_loss = [device_loss.item() for device_loss in gather_step_tensor(loss)]
                    self.valid_step_loss += sum(step_loss) / self.accelerator.num_processes

                    pred_idx = get_idx_rank(scores, labels)
                    step_pred = torch.cat(gather_step_tensor(pred_idx), dim=0)
                    self.metric.update(step_pred.tolist())

                    if (step + 1) % 10 == 0:
                        valid_avg_loss = self.valid_step_loss / (step + 1)
                        valid_epoch.set_postfix({"step": step, "loss": "{:.2f}".format(valid_avg_loss)})

                self.valid_mean_loss = self.valid_step_loss / len(self.valid_dataloader)
                self.valid_mrr = self.metric.compute()
                msg = f"Epoch {epoch_idx + 1}, loss: {self.valid_mean_loss}, mrr: {self.valid_mrr}"

                if self.accelerator.is_main_process:
                    self.logger.info(msg) if self.logger else print(msg)
            # save model by step

    def save_model(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process and self.args.output_dir is not None:
            msg = f"Save model checkpoint..."
            self.logger.info(msg)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self.accelerator.save(unwrapped_model, self.args.output_dir + "pytorch_model.bin")
