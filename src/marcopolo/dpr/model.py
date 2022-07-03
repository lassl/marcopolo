from transformers import AutoModel, AutoConfig
import torch.nn as nn

class DPR(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        # self.backbone.init_weights()

    # @autocast()
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids= input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output

        return pooler_output