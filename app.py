import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers import FluxControlNetPipeline
from io import BytesIO
import base64
import inferless
from pydantic import BaseModel, Field
from typing import Optional

@inferless.request
class RequestObjects(BaseModel):
        prompt: str = Field(default="Soft and Natural")
        negative_prompt: str = Field(default="No missing lines")
        controlnet_conditioning_scale:float = Field(default=0.7)
        num_inference_steps:int = Field(default=50)
        guidance_scale:float = Field(default=7.5)
        image_url: str = Field(default="https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/examples/input.jpg")
        upscale_factor: int = Field(default=4)
        max_dimension: int = Field(default=2048)
        seed:int = Field(default=42)

@inferless.response
class ResponseObjects(BaseModel):
        generated_image_base64: str = Field(default='Test output')

class InferlessPythonModel:
    @staticmethod
    def process_input_image(input_image, upscale_factor, max_dimension):    
        MAX_PIXEL_BUDGET = max_dimension * max_dimension
        w, h = input_image.size
        w_original, h_original = w, h
        aspect_ratio = w / h
    
        was_resized = False
    
        if w * h * upscale_factor**2 > MAX_PIXEL_BUDGET:
            input_image = input_image.resize(
                (
                    int(aspect_ratio * MAX_PIXEL_BUDGET**0.5 // upscale_factor),
                    int(MAX_PIXEL_BUDGET**0.5 // aspect_ratio // upscale_factor),
                )
            )
            was_resized = True
    
        # resize to multiple of 8
        w, h = input_image.size
        w = w - w % 8
        h = h - h % 8
        
        return input_image.resize((w, h)), w_original, h_original, was_resized

    def initialize(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device 
        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet_model = 'jasperai/Flux.1-dev-Controlnet-Upscaler'
          
        controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16, use_safetensors=True)
        pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16, use_safetensors=True)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        self.model = pipe
        self.model.to(device) # move model to device

    def infer(self, request: RequestObjects) -> ResponseObjects:
        control_image = load_image(
        request.image_url
        )        
        control_image, w_original, h_original, was_resized = self.process_input_image(
        control_image, request.upscale_factor, request.max_dimension
        )
        
        w, h = control_image.size
        control_image = control_image.resize((w * upscale_factor, h * upscale_factor))
        
        generator = torch.Generator().manual_seed(request.seed)
        
        output_image = self.model(
        prompt=request.prompt, 
        prompt_2=request.negative_prompt, 
        control_image=control_image,
        controlnet_conditioning_scale=request.controlnet_conditioning_scale,
        num_inference_steps=request.num_inference_steps, 
        guidance_scale=request.guidance_scale,
        height=control_image.size[1],
        width=control_image.size[0],
        generator=generator,
        ).images[0]
        
        output_image = output_image.resize((w_original * upscale_factor, h_original * upscale_factor))
        buff = BytesIO()
        output_image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()

        generateObject = ResponseObjects(generated_image_base64 = img_str)
        return generateObject

    def finalize(self):
        self.model = None