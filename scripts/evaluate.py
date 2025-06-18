# scripts/evaluate.py
"""
evaluate.py

This script evaluates a fine-tuned DistilBERT model on the IMDb test dataset.
It computes accuracy and weighted F1-score using sklearn, and saves the results to `results.json`.

Assumes the model and tokenizer are saved under ./models/distilbert-sentiment.
"""
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import torch
import json

# Load test data
dataset = load_dataset("imdb")
test_dataset = dataset["test"].select(range(10))

# Load model and tokenizer
model_path = "./models/distilbert-sentiment"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Tokenize test set
def tokenize(batch):
    """
    Tokenizes a batch of text samples using the loaded tokenizer.

    Args:
        batch (dict): A dictionary with a "text" key containing list of strings.

    Returns:
        dict: A dictionary with tokenized input_ids and attention_mask.
    """
    return tokenizer(batch["text"], padding=True, truncation=True)

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# Predict
model.eval()
predictions = []
labels = test_dataset["label"]

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds = torch.argmax(outputs.logits, axis=1)
        predictions.extend(preds.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average="weighted")

# Save result
results = {"accuracy": accuracy, "f1_score": f1}
with open("results.json", "w") as f:
    json.dump(results, f)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
