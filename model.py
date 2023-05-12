import os
from random import random, sample

import evaluate
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


class TextDataset(Dataset):
    def __init__(self, text_file, block_size=256):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.block_size = block_size
        self.pad = self.tokenizer.encode(self.tokenizer.eos_token)[0]
        self.data = {"train": [], "validation": []}
        self.data["train"] = self.generate_dateset(text_file)
        self.data["validation"] = self.generate_dateset(text_file, eval=True, split=0.1)

    def generate_dateset(self, text_file, eval=False, split=0.1):
        data = []
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().split("<BREAK>")
            if eval:
                text = sample(text, int(len(text) * split))

        input_ids = []
        attention_mask = []
        for t in text:
            input_ids.extend(self.tokenizer.encode(t))
            attention_mask.extend([1 for i in range(len(input_ids))])
        input_ids = self.pad_list(input_ids)
        attention_mask = self.pad_list(attention_mask)
        id_chunks = list(self.chunks(input_ids, self.block_size))
        attm_chunks = list(self.chunks(attention_mask, self.block_size))

        for id, mask in zip(id_chunks, attm_chunks):
            data.append(
                {
                    "input_ids": torch.tensor(id),
                    "attention_mask": torch.tensor(mask),
                    "labels": torch.tensor(id.copy()),
                }
            )
        return data

    def pad_list(self, input_ids):
        padding = self.block_size - (len(input_ids) % self.block_size)
        _ = [input_ids.append(self.pad) for _ in range(padding)]
        return input_ids

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def group_texts(examples, block_size=512):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def train_lm(resume_from_checkpoint=False):
    # Create Tokenizer and Model objects
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    block_size = 512
    dataset = TextDataset("corpus.txt", block_size=block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-8)

    train_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=block_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_collator,
    )
    eval_dataloader = DataLoader(
        dataset["validation"],
        batch_size=block_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_collator,
    )

    torch.cuda.empty_cache()

    model_checkpoint = "ethical"
    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-finetuned-gpt2",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=15,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=train_collator,
        optimizers=(optimizer, None),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
