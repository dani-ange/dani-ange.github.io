from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
dataset = load_dataset("imdb")

# Split the original train set into train/validation
train_valid_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = train_valid_split["train"].select(range(10))
valid_dataset = train_valid_split["test"].select(range(10))
test_dataset = dataset["test"].select(range(10))  # use this only for final testing

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    """
    Tokenizes a batch of text data using the DistilBERT tokenizer.

    Args:
        batch (dict): A dictionary containing a "text" field with a list of strings.

    Returns:
        dict: A dictionary with tokenized inputs, including input IDs and attention masks.
    """
    return tokenizer(batch["text"], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for the model.

    Args:
        eval_pred (tuple): A tuple containing model logits and true labels.

    Returns:
        dict: A dictionary with accuracy and F1 score.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

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
    compute_metrics=compute_metrics,
)

# Train and save model
trainer.train()
model.save_pretrained("./models/distilbert-sentiment")
tokenizer.save_pretrained("./models/distilbert-sentiment")
