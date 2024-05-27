"""
This is a simple chat demo using CogVLM2 model in ChainLit.
"""
import os
import dataclasses
from typing import List
from PIL import Image
import chainlit as cl
from chainlit.input_widget import Slider
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from huggingface_hub.inference._generated.types import TextGenerationStreamOutput, TextGenerationStreamOutputToken
import threading
import torch

MODEL_PATH = 'THUDM/cogvlm2-llama3-chat-19B'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

quant = int(os.environ.get('QUANT', 0))
if 'int4' in MODEL_PATH:
    quant = 4
print(f'Quant = {quant}')

# Load the model
if quant == 4:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        load_in_4bit=True,
        low_cpu_mem_usage=True
    ).eval()
elif quant == 8:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        load_in_8bit=True,  # Assuming transformers support this argument; check documentation if not
        low_cpu_mem_usage=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True
    ).eval().to(DEVICE)

@cl.on_chat_start
def on_chat_start():
    print("Welcome use CogVLM2 chat demo")


async def get_response(query, history, gen_kwargs, images=None):
    if images is None:
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
            images=images[-1:],  # only use the last image, CogVLM2 only support one image
            template_version='chat'
        )

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if images is not None else None,
    }

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs['streamer'] = streamer
    gen_kwargs = {**gen_kwargs, **inputs}
    with torch.no_grad():
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        for next_text in streamer:
            yield TextGenerationStreamOutput(
                index=0,
                token=TextGenerationStreamOutputToken(
                    id=0,
                    logprob=0,
                    text=next_text,
                    special=False,
                )
            )


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    roles: List[str]
    messages: List[List[str]]
    version: str = "Unknown"

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_prompt(self):
        if not self.messages:
            return None, []

        last_role, last_msg = self.messages[-2]
        if isinstance(last_msg, tuple):
            query, _ = last_msg
        else:
            query = last_msg

        history = []
        for role, msg in self.messages[:-2]:
            if isinstance(msg, tuple):
                text, _ = msg
            else:
                text = msg

            if role == "USER":
                history.append((text, ""))
            else:
                if history:
                    history[-1] = (history[-1][0], text)

        return query, history

    def get_images(self):
        for role, msg in reversed(self.messages):
            if isinstance(msg, tuple):
                msg, image = msg
                if image is None:
                    continue
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                width, height = image.size
                if width > 1344 or height > 1344:
                    max_len = 1344
                    aspect_ratio = width / height
                    if width > height:
                        new_width = max_len
                        new_height = int(new_width / aspect_ratio)
                    else:
                        new_height = max_len
                        new_width = int(new_height * aspect_ratio)
                    image = image.resize((new_width, new_height))
                return [image]
        return None

    def copy(self):
        return Conversation(
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            version=self.version,
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "roles": self.roles,
                "messages": [
                    [x, y[0] if type(y) is tuple else y] for x, y in self.messages
                ],
            }
        return {
            "roles": self.roles,
            "messages": self.messages,
        }


default_conversation = Conversation(
    roles=("USER", "ASSISTANT"),
    messages=()
)


async def request(conversation: Conversation, settings):
    gen_kwargs = {
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "max_new_tokens": int(settings["max_token"]),
        "top_k": int(settings["top_k"]),
        "do_sample": True,
    }
    query, history = conversation.get_prompt()
    images = conversation.get_images()

    chainlit_message = cl.Message(content="", author="CogVLM2")
    text = ""
    async for response in get_response(query, history, gen_kwargs, images):
        output = response.token.text
        text += output
        conversation.messages[-1][-1] = text
        await chainlit_message.stream_token(text, is_sequence=True)

    await chainlit_message.send()
    return conversation


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Slider(id="temperature", label="Temperature", initial=0.5, min=0.01, max=1, step=0.05),
            Slider(id="top_p", label="Top P", initial=0.7, min=0, max=1, step=0.1),
            Slider(id="top_k", label="Top K", initial=5, min=0, max=50, step=1),
            Slider(id="max_token", label="Max output tokens", initial=2048, min=0, max=8192, step=1),
        ]
    ).send()

    conversation = default_conversation.copy()

    cl.user_session.set("conversation", conversation)
    cl.user_session.set("settings", settings)


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)


@cl.on_message
async def main(message: cl.Message):
    image = next(
        (
            Image.open(file.path)
            for file in message.elements or []
            if "image" in file.mime and file.path is not None
        ),
        None,
    )

    conv = cl.user_session.get("conversation")  # type: Conversation
    settings = cl.user_session.get("settings")

    text = message.content

    conv_message = (text, image)
    conv.append_message(conv.roles[0], conv_message)
    conv.append_message(conv.roles[1], None)

    conv = await request(conv, settings)
    cl.user_session.set("conversation", conv)
