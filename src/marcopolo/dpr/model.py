import torch
import os
import torch.nn as nn
from transformers import AutoModel


class DualEncoder(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.q_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.c_encoder = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, is_query=False, **kwargs):
        output = self.q_encoder(**kwargs) if is_query else self.c_encoder(**kwargs)
        output = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state[:, 0]
        return output
    
    def save_pretrained(self, save_directory, is_main_process=True, save_function=torch.save):
        q_encoder_path = os.path.join(save_directory, "q_encoder")
        c_encoder_path = os.path.join(save_directory, "c_encoder")
        self.q_encoder.save_pretrained(q_encoder_path, is_main_process=is_main_process, save_function=save_function)
        self.c_encoder.save_pretrained(c_encoder_path, is_main_process=is_main_process, save_function=save_function)


    def from_pretrained(self, pretrained_model_name_or_path):
        if os.path.exists(pretrained_model_name_or_path):
            q_encoder_path = os.path.join(pretrained_model_name_or_path, "q_encoder")
            c_encoder_path = os.path.join(pretrained_model_name_or_path, "c_encoder")
            self.q_encoder = AutoModel.from_pretrained(q_encoder_path)
            self.c_encoder = AutoModel.from_pretrained(c_encoder_path)
        else:
            self.q_encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)
            self.c_encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)