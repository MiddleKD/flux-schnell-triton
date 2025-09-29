import torch
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils import USE_PEFT_BACKEND
from typing import Union, List, Optional
import asyncio

async_runner = asyncio.Runner()

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

    def _tokenize_clip_prompt(prompt):
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

    def _tokenize_t5_prompt(prompt):
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

    async def _do_work(input_ids_clip, input_ids_t5):
        # text encode
        pooled_prompt_embeds, prompt_embeds = await asyncio.gather(
            _get_clip_prompt_embeds(input_ids_clip),
            _get_t5_prompt_embeds(input_ids_t5)
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

        # tokenize
        clip_input_ids = _tokenize_clip_prompt(prompt)
        t5_input_ids = _tokenize_t5_prompt(prompt_2)

        pooled_prompt_embeds, prompt_embeds = async_runner.run(
            _do_work(clip_input_ids, t5_input_ids)
        )
    
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
