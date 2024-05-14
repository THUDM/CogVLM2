from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import json
import os
import random
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


class ConversationDataset(Dataset):
    def __init__(self, root_dir, tokenizer, model, config, torch_type=torch.float16, device='cuda'):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels_en')
        self.filenames = os.listdir(self.image_dir)
        self.output_length = 2400
        self.input_length = 300
        self.device = device
        self.torch_type = torch_type

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def custom_collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            # For list[tensor]] structures
            if isinstance(batch[0][key], list):
                batched_data[key] = [batch_item[key] for batch_item in batch]
            # For tensor structures
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                raise ValueError("Unsupported datatype in custom collate_fn")

        return batched_data

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.filenames[idx])
        label_name = os.path.join(self.label_dir, self.filenames[idx].replace('.jpg', '.json'))

        image = Image.open(img_name).convert('RGB')
        with open(label_name, 'r') as f:
            label_data = json.load(f)

        num_rounds = len(label_data["conversations"]) // 2
        # sampled round to train
        sampled_round_id = random.randint(0, num_rounds - 1)
        # sampled_rounds-1 -th rounds are used as history
        history = [(label_data["conversations"][(sampled_round_id - 1) * 2]["content"],
                    label_data["conversations"][(sampled_round_id - 1) * 2 + 1]["content"])] if (
                sampled_round_id > 0 and random.random() > 0.5) else None
        # the last sampled round is used as query & response
        query = label_data["conversations"][sampled_round_id * 2]["content"]
        response = label_data["conversations"][sampled_round_id * 2 + 1]["content"]

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            history=history,
            images=[image],
            answer=response
        )

        def pad_to_len(unpadded_tensor, pad_to_length, pad_value=0):
            if len(unpadded_tensor) >= pad_to_length:
                # todo: better way to handle this
                return unpadded_tensor[:pad_to_length]
            return torch.cat(
                (unpadded_tensor,
                 torch.full([pad_to_length - len(unpadded_tensor)],
                            fill_value=pad_value,
                            dtype=unpadded_tensor.dtype,
                            device=unpadded_tensor.device)), dim=0)

        input_data['attention_mask'] = pad_to_len(input_data['attention_mask'], self.output_length, pad_value=0)
        input_data['token_type_ids'] = pad_to_len(input_data['token_type_ids'], self.output_length, pad_value=0)
        input_data['input_ids'] = pad_to_len(input_data['input_ids'], self.output_length,
                                             pad_value=self.tokenizer.pad_token_id)
        input_data['labels'] = pad_to_len(input_data['labels'], self.output_length, pad_value=-100)

        for data_key in input_data:
            if data_key in ['images']:
                input_data[data_key] = [data.to(self.device).to(self.torch_type) for data in input_data[data_key]]
            else:
                input_data[data_key] = input_data[data_key].to(self.device)

        return input_data


def finetune():
    lr = 1e-5
    num_epochs = 5
    batch_size = 1
    torch_type = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(
        "THDUM/CogVLM2",
        trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        "THUDM/CogVLM2",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "THUDM/CogVLM2",
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        trust_remote_code=True).to(device)

    # Instantiate your Dataset
    dataset = ConversationDataset(
        root_dir='THUDM/CogVLM-SFT-311K/llava_instruction_single_conversation_formate',
        tokenizer=tokenizer,
        model=model,
        config=config,
        torch_type=torch_type,
        device=device
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=dataset.custom_collate_fn)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.custom_collate_fn)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        target_modules=
        [
            # "query_key_value", # vit
            "language_expert_query_key_value"  # language's attention
        ],
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.to(device).half()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                images=batch['images'],
                labels=batch['labels']
            )

            # language_output = outputs
            loss = outputs.loss
            print("loss: ", loss)

            # chek nan
            # if torch.isnan(loss):
            #     print(f"NaN detected at step {step}")
            #     print("Outputs logits:", outputs.logits)
            #     print("Gradient norms:",
            #           {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None})
            #     break

            total_loss += loss.detach().float()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                                       skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    model.save_pretrained("output_dir")


if __name__ == "__main__":
    finetune()
