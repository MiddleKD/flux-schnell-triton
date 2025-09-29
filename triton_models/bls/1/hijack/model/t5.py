import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

class DummyT5(PreTrainedModel):
    def __init__(self, model_name, subfolder):
        config = PretrainedConfig.from_pretrained(model_name, subfolder=subfolder)
        super().__init__(config)
        self.dummy_weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch_size = input_ids.shape[0]
        return torch.zeros((batch_size, 512, 4096))
