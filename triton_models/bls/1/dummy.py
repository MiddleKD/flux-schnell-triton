import torch
from torch import nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models import CacheMixin
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils import USE_PEFT_BACKEND
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from typing import Union, List, Optional, Dict, Any
import asyncio

def encode_prompt(
    self,
    prompt: Union[str, List[str]],
    prompt_2: Optional[Union[str, List[str]]] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 512,
    lora_scale: Optional[float] = None,
):

    async def _tokenize_clip_prompt(prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        return text_inputs.input_ids

    async def _tokenize_t5_prompt(prompt):
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        return text_inputs.input_ids

    async def _get_clip_prompt_embeds(input_ids):
        return self.text_encoder(input_ids, output_hidden_states=False)
    
    async def _get_t5_prompt_embeds(input_ids):
        return self.text_encoder_2(input_ids, output_hidden_states=False)

    async def _do_work(prompt, prompt_2):
        # tokenize
        clip_input_ids, t5_input_ids = await asyncio.gather(
            _tokenize_clip_prompt(prompt),
            _tokenize_t5_prompt(prompt_2)
        )

        # text encode
        pooled_prompt_embeds, prompt_embeds = await asyncio.gather(
            _get_clip_prompt_embeds(clip_input_ids),
            _get_t5_prompt_embeds(t5_input_ids)
        )

        return pooled_prompt_embeds, prompt_embeds
        
    device = device or self._execution_device

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
        self._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if self.text_encoder is not None and USE_PEFT_BACKEND:
            scale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
            scale_lora_layers(self.text_encoder_2, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        pooled_prompt_embeds, prompt_embeds = asyncio.run(_do_work(prompt, prompt_2))
    
    if self.text_encoder is not None:
        if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

    if self.text_encoder_2 is not None:
        if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder_2, lora_scale)

    dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


class DummyClip(PreTrainedModel):
    def __init__(self, model_name, subfolder):
        config = PretrainedConfig.from_pretrained(model_name, subfolder=subfolder)
        super().__init__(config)
        self.dummy_weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch_size = input_ids.shape[0]
        return torch.zeros((batch_size, 768))


class DummyT5(PreTrainedModel):
    def __init__(self, model_name, subfolder):
        config = PretrainedConfig.from_pretrained(model_name, subfolder=subfolder)
        super().__init__(config)
        self.dummy_weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch_size = input_ids.shape[0]
        return torch.zeros((batch_size, 512, 4096))


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


class DummyVAE(ModelMixin):
    def __init__(self, model_name, subfolder):
        super().__init__()
        self.config = PretrainedConfig.from_pretrained(model_name, subfolder=subfolder)
        self.dummy_weight = nn.Parameter(torch.zeros(1))

    def decode(self, latents, return_dict=False):
        batch_size = latents.shape[0]
        return [torch.zeros((batch_size, 3, 1024, 1024))]
