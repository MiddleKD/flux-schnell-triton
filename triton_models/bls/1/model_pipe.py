
import json
import logging
import numpy as np
import traceback
import math
import sys
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)


import diffusers
from diffusers import FluxPipeline
import torch
import hijack

class TritonPythonModel:
    def initialize(self, args: Dict[str, Any]) -> None:

        model_name = "black-forest-labs/FLUX.1-schnell"

        clip = hijack.DummyClip(model_name, subfolder="text_encoder")
        t5 = hijack.DummyT5(model_name, subfolder="text_encoder_2")
        dit = hijack.DummyDIT(model_name, subfolder="transformer")
        vae = hijack.DummyVAE(model_name, subfolder="vae")

        self.pipeline = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="balanced",
            text_encoder=clip,
            text_encoder_2=t5,
            transformer=dit,
            vae=vae,
            safety_checker=None
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.encode_prompt = hijack.encode_prompt.__get__(self.pipeline, type(self.pipeline))



    def execute(self, requests: List) -> List:
        
        prompt = ["a tiny astronaut hatching from an egg on the moon", "a tiny astronaut hatching from an egg on the moon"]
        images = self.pipeline(
            prompt,
            guidance_scale=3.5,
            height=1024,
            width=1024,
            num_inference_steps=4,
            output_type="pt"
        )

        for image in images.images:
            print(image.shape)


    def finalize(self) -> None:
        """Clean up resources"""
        logger.info("BLS Orchestrator finalized")


# Test execution capability (direct execution for development)
if __name__ == "__main__":
    model = TritonPythonModel()
    model.initialize({})
    model.execute([])