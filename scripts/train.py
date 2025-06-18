from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# Load dataset
dataset = load_dataset("imdb")

# Split the original train set into train/validation
train_valid_split = dataset["train"].train_test_split(test_size=0.2, seed=42).select(range(10))
train_dataset = train_valid_split["train"]
valid_dataset = train_valid_split["test"]
test_dataset = dataset["test"].select(range(5))  # use this only for final testing

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="epoch",
    report_to="none"  # Disable wandb reporting
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Train and save model
trainer.train()
model.save_pretrained("./models/distilbert-sentiment")
tokenizer.save_pretrained("./models/distilbert-sentiment")