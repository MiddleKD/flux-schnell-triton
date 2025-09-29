import torch
from torch import nn
from diffusers.models.modeling_utils import ModelMixin
from transformers.modeling_utils import PretrainedConfig

class DummyVAE(ModelMixin):
    def __init__(self, model_name, subfolder):
        super().__init__()
        self.config = PretrainedConfig.from_pretrained(model_name, subfolder=subfolder)
        self.dummy_weight = nn.Parameter(torch.zeros(1))

    def decode(self, latents, return_dict=False):
        batch_size = latents.shape[0]
        return [torch.zeros((batch_size, 3, 1024, 1024))]
