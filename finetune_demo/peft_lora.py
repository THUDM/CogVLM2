from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import json
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


class ConversationDataset(Dataset):
    def __init__(self,
                 root_dir,
                 tokenizer,
                 model,
                 torch_type=torch.bfloat16,
                 device='cuda',
                 input_length=75,
                 output_length=75):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels_en')
        self.filenames = os.listdir(self.image_dir)
        self.input_length = input_length
        self.output_length = output_length
        self.device = device
        self.torch_type = torch_type
        self.padding_len = 2306
        self.max_length = self.input_length + self.output_length + self.padding_len

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def custom_collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched_data[key] = [batch_item[key] for batch_item in batch]
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
        sampled_round_id = random.randint(0, num_rounds - 1)
        history = [(label_data["conversations"][(sampled_round_id - 1) * 2]["content"],
                    label_data["conversations"][(sampled_round_id - 1) * 2 + 1]["content"])] if (
                sampled_round_id > 0 and random.random() > 0.5) else None
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
                return unpadded_tensor[:pad_to_length]
            return torch.cat(
                (unpadded_tensor,
                 torch.full([pad_to_length - len(unpadded_tensor)],
                            fill_value=pad_value,
                            dtype=unpadded_tensor.dtype,
                            device=unpadded_tensor.device)), dim=0)

        input_data['input_ids'] = pad_to_len(
            input_data['input_ids'],
            self.max_length,
            pad_value=128002,
        )

        input_data['attention_mask'] = pad_to_len(
            input_data['attention_mask'],
            self.max_length,
            pad_value=0
        )
        input_data['token_type_ids'] = pad_to_len(
            input_data['token_type_ids'],
            self.max_length,
            pad_value=0
        )

        input_data['labels'] = pad_to_len(
            input_data['labels'],
            self.max_length,
            pad_value=-100
        )

        for data_key in input_data:
            if data_key in ['images']:
                input_data[data_key] = [data.to(self.device).to(self.torch_type) for data in input_data[data_key]]
            else:
                input_data[data_key] = input_data[data_key].to(self.device)

        print(input_data["input_ids"].size())

        return input_data


def finetune():
    lr = 1e-10
    num_epochs = 5
    batch_size = 1
    torch_type = torch.bfloat16
    model_path = "CogVLM2"
    dataset_path = 'llava_instruction_single_conversation_formate'

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch_type,
        trust_remote_code=True).cuda()

    dataset = ConversationDataset(
        root_dir=dataset_path,
        tokenizer=tokenizer,
        model=model,
        torch_type=torch_type
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.custom_collate_fn,
    )

    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.custom_collate_fn,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        target_modules=
        [
            # "query_key_value", # vit
            # "vision_expert_query_key_value", # vit
            # "language_expert_query_key_value"  # language's attention
            # "vision_expert_query_key_value",
            # "vision_expert_dense",
            "language_expert_query_key_value",
            # "language_expert_dense"
        ],
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
            outputs = model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                images=batch['images'],
                labels=batch['labels']
            )

            loss = outputs.loss
            print(f"{loss=}")
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(
                    torch.argmax(
                        outputs.logits, -1).detach().cpu().numpy(),
                    kip_special_tokens=True
                )
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoint_epoch_{epoch}.pth.tar')

    # 最终保存模型
    model.save_pretrained("output_dir")


if __name__ == "__main__":
    finetune()
