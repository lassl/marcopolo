from transformers import AutoModel


class DPR:
    def __init__(self, model_name_or_path: str):
        self.q_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.c_encoder = AutoModel.from_pretrained(model_name_or_path)
