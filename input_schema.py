INPUT_SCHEMA = {
    "image_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/examples/input.jpg"]
    },
    "upscale_factor": {
        'datatype': 'INT64',
        'required': True,
        'shape': [1],
        'example': [4]
    },
    "max_dimension": {
        'datatype': 'INT64',
        'required': True,
        'shape': [1],
        'example': [2048]
    },
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Soft and Natural"]
    },
    "negative_prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["No missing lines"]
    },
    "controlnet_conditioning_scale": {
        'datatype': 'FP64',
        'required': True,
        'shape': [1],
        'example': [0.7]
    },
    "num_inference_steps": {
        'datatype': 'INT64',
        'required': True,
        'shape': [1],
        'example': [50]
    },
    "guidance_scale": {
        'datatype': 'FP64',
        'required': True,
        'shape': [1],
        'example': [7.5]
    },
    "seed": {
        'datatype': 'INT64',
        'required': True,
        'shape': [1],
        'example': [42]
    }
}
