import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers import FluxControlNetPipeline
from huggingface_hub import login
import os
from io import BytesIO
import base64


class InferlessPythonModel:
    @staticmethod
    def process_input_image(input_image, upscale_factor, max_dimension):
    
        MAX_PIXEL_BUDGET = max_dimension * max_dimension
        
        w, h = input_image.size
        w_original, h_original = w, h
        aspect_ratio = w / h
    
        was_resized = False
    
        if w * h * upscale_factor**2 > MAX_PIXEL_BUDGET:
            print(
                f"Requested output image is too large ({w * upscale_factor}x{h * upscale_factor}). Resizing to ({int(aspect_ratio * MAX_PIXEL_BUDGET ** 0.5 // upscale_factor), int(MAX_PIXEL_BUDGET ** 0.5 // aspect_ratio // upscale_factor)}) pixels.",
                flush=True
            )
            print(
                f"Requested output image is too large ({w * upscale_factor}x{h * upscale_factor}). Resizing input to ({int(aspect_ratio * MAX_PIXEL_BUDGET ** 0.5 // upscale_factor), int(MAX_PIXEL_BUDGET ** 0.5 // aspect_ratio // upscale_factor)}) pixels budget.",
                flush=True
            )
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
        huggingface_token = os.getenv("HUGGINFACE_TOKEN")
        #LOGIN TO USE THE GATED FLUX DEV MODEL
        login(token=huggingface_token, add_to_git_credential=True) 
        
        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet_model = 'jasperai/Flux.1-dev-Controlnet-Upscaler'
          
        controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16, use_safetensors=True)
        pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16, use_safetensors=True)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
        self.model = pipe
        self.model.to(device) # move model to device


    def infer(self, inputs):
        control_image = load_image(
        inputs["image_url"]
        )
        upscale_factor = int(inputs["upscale_factor"])
        max_dimension = int(inputs["max_dimension"])
        
        control_image, w_original, h_original, was_resized = self.process_input_image(
        control_image, upscale_factor, max_dimension
        )
        
        print("was_resized >>> ", was_resized,flush=True)
        
        # rescale with upscale factor
        w, h = control_image.size
        control_image = control_image.resize((w * upscale_factor, h * upscale_factor))
        
        print("control_image after resizing >>> ", control_image,flush=True)
        
        # Create a seed
        seed = int(inputs["seed"])
        generator = torch.Generator().manual_seed(seed)
        
        output_image = self.model(
        prompt=inputs["prompt"], 
        prompt_2=inputs["negative_prompt"], 
        control_image=control_image,
        controlnet_conditioning_scale=float(inputs["controlnet_conditioning_scale"]),
        num_inference_steps=int(inputs["num_inference_steps"]), 
        guidance_scale=float(inputs["guidance_scale"]),
        height=control_image.size[1],
        width=control_image.size[0],
        generator=generator,
        ).images[0]
        
        if was_resized:
            print(
            f"Resizing output image to targeted {w_original * upscale_factor}x{h_original * upscale_factor} size.",flush=True
            )
        
        output_image = output_image.resize((w_original * upscale_factor, h_original * upscale_factor))
        buff = BytesIO()
        output_image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()

        return {"generated_image_base64": img_str}
          
    def finalize(self):
        self.model = None
