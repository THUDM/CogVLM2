# Prediction interface for Cog ⚙️
# https://cog.run/python


import os
import time
import subprocess
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from cog import BasePredictor, Input, Path

MODEL_CACHE = "model_cache_image"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/THUDM/CogVLM2/{MODEL_CACHE}.tar"
)
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

TORCH_TYPE = torch.bfloat16
DEVICE = "cuda:0"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # model_id: THUDM/cogvlm2-llama3-chat-19B, use 8 bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            low_cpu_mem_usage=True,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CACHE, trust_remote_code=True
        )

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt", default="Describe this image."),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.9,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.7,
            ge=0.0,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=2048,
            ge=0,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        image = Image.open(str(input_image)).convert("RGB")

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer, query=prompt, images=[image], template_version="chat"
        )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
            "images": (
                [[input_by_model["images"][0].to(DEVICE).to(TORCH_TYPE)]]
                if image is not None
                else None
            ),
        }
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": 128002,
            # "top_k": 1,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
