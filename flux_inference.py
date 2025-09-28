import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

torch.cuda.empty_cache()

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    text_encoder_2=text_encoder,
    transformer=transformer,
    dtype=torch.float16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt, guidance_scale=3.5, height=768, width=768, num_inference_steps=4).images[0]
image.save("flux-schnell.png")