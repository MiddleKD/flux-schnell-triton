import torch
from torch import nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models import CacheMixin
from transformers.modeling_utils import PretrainedConfig
from typing import Optional, Dict, Any

class DummyDIT(ModelMixin, CacheMixin):
    def __init__(self, model_name, subfolder):
        super().__init__()
        self.config = PretrainedConfig.from_pretrained(model_name, subfolder=subfolder)
        self.dummy_weight = nn.Parameter(torch.zeros(1))

    def forward(self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
        ):
        batch_size = hidden_states.shape[0]
        return torch.zeros((batch_size, 4096, 64))
