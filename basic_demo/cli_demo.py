"""
This is a demo for using CogVLM2 in CLI using Single GPU.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
for multi-GPU, please use cli_demo_multi_gpus.py
"""
import torch
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

# Argument parser
parser = argparse.ArgumentParser(description="CogVLM2 CLI Demo")
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
args = parser.parse_args()

if 'int4' in MODEL_PATH:
    args.quant = 4

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# Check GPU memory
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 48 * 1024 ** 3 and not args.quant:
    print("GPU memory is less than 48GB. Please use cli_demo_multi_gpus.py or pass `--quant 4` or `--quant 8`.")
    exit()

# Load the model
if args.quant == 4:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        low_cpu_mem_usage=True
    ).eval()
elif args.quant == 8:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        low_cpu_mem_usage=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True
    ).eval().to(DEVICE)

while True:
    image_path = input("image path >>>>> ")
    if image_path == '':
        print('You did not enter image path, the following will be a plain text conversation.')
        image = None
        text_only_first_query = True
    else:
        image = Image.open(image_path).convert('RGB')

    history = []

    while True:
        query = input("Human:")
        if query == "clear":
            break

        if image is None:
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                template_version='chat'
            )
        else:
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version='chat'
            )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\nCogVLM2:", response)
        history.append((query, response))
