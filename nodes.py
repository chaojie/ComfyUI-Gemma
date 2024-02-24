import os
import folder_paths
import random

import numpy as np
import torch

import kagglehub
#kagglehub.login()

comfy_path = os.path.dirname(folder_paths.__file__)
# Choose variant and machine type
#VARIANT = '2b-it' #@param ['2b', '2b-it', '7b', '7b-it', '7b-quant', '7b-it-quant']
#MACHINE_TYPE = 'cuda' #@param ['cuda', 'cpu']

# Load model weights
#weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')

class GemmaLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "VARIANT": (['2b', '2b-it', '7b', '7b-it', '7b-quant', '7b-it-quant'],{"default":"7b-it-quant"}),
                "MACHINE_TYPE": (['cuda', 'cpu'],{"default":"cuda"}),
                "weights_dir": ("STRING",{"default":"/home/admin/.cache/kagglehub/models/google/gemma/pyTorch/7b-it-quant/2"}),
            },
        }

    RETURN_TYPES = ("GemmaModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Gemma"

    def run(self,VARIANT,MACHINE_TYPE,weights_dir):
        # Ensure that the tokenizer is present
        tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
        print(tokenizer_path)
        assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

        # Ensure that the checkpoint is present
        ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
        assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'   

        from .gemma.config import get_config_for_7b, get_config_for_2b
        from .gemma.model import GemmaForCausalLM

        # Set up model config.
        model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
        model_config.tokenizer = tokenizer_path
        model_config.quant = 'quant' in VARIANT

        # Instantiate the model and load the weights.
        torch.set_default_dtype(model_config.get_dtype())
        device = torch.device(MACHINE_TYPE)
        model = GemmaForCausalLM(model_config)
        model.load_weights(ckpt_path)
        model = model.to(device).eval()

        return (model,)

class GemmaRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("GemmaModel",),
                "MACHINE_TYPE": (['cuda', 'cpu'],{"default":"cuda"}),
                "prompt": ("STRING",{"default":""}),
                "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "Gemma"

    def run(self,model,MACHINE_TYPE,prompt,seed):
        device = torch.device(MACHINE_TYPE)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Generate sample
        result=model.generate(
            prompt,
            device=device,
            output_len=60,
        )

        return (f'{result}',)

NODE_CLASS_MAPPINGS = {
    "GemmaLoader":GemmaLoader,
    "GemmaRun":GemmaRun,
}
