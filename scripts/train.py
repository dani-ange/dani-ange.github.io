# scripts/train.py

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# Load dataset
dataset = load_dataset("imdb")
dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.2)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_dataset = dataset["train"].map(tokenize, batched=True)
test_dataset = dataset["test"].map(tokenize, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train and save model
trainer.train()
model.save_pretrained("./models/distilbert-sentiment")
tokenizer.save_pretrained("./models/distilbert-sentiment")
