import torch
from torch import nn
from transformers import PreTrainedModel


class Biencoder(nn.Module):
    def __init__(
        self,
        question_model: PreTrainedModel,
        ctx_model: PreTrainedModel,
        question_batch_nm: str = "question_batch",
        context_batch_nm: str = "context_batch",
    ):
        super().__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.question_batch_nm = question_batch_nm
        self.context_batch_nm = context_batch_nm

    def forward(self, batch):
        q_pooled_output = self.question_model(**batch[self.question_batch_nm]).last_hidden_state[:, 0, :]
        ctx_pooled_output = self.ctx_model(**batch[self.context_batch_nm]).last_hidden_state[:, 0, :]
        return q_pooled_output, ctx_pooled_output
