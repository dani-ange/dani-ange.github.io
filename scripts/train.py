from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import json
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
    num_train_epochs=3,
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
# --- Save training logs to log.json ---
log_history = trainer.state.log_history  # list of dicts, one per log step
with open("docs/log.json", "w") as f:
    json.dump(log_history, f, indent=2)

print("✅ Training complete. Logs saved to log.json.")

html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Training Log</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans p-6">
  <div class="max-w-4xl mx-auto bg-white rounded shadow p-6">
    <h1 class="text-2xl font-bold mb-4">📈 DistilBERT Training Log</h1>
    <table class="min-w-full table-auto border border-gray-300">
      <thead class="bg-gray-200">
        <tr>
          <th class="px-4 py-2 text-left">Step</th>
          <th class="px-4 py-2 text-left">Epoch</th>
          <th class="px-4 py-2 text-left">Loss</th>
          <th class="px-4 py-2 text-left">Eval Accuracy</th>
          <th class="px-4 py-2 text-left">Eval F1</th>
        </tr>
      </thead>
      <tbody id="log-body" class="bg-white">
        <!-- Rows will be filled by JS -->
      </tbody>
    </table>
  </div>

  <script>
    fetch("log.json")
      .then(res => res.json())
      .then(data => {{
        const tbody = document.getElementById("log-body");
        data.forEach(entry => {{
          const row = document.createElement("tr");
          row.className = "border-t border-gray-200";

          const step = entry.step ?? "";
          const epoch = entry.epoch ?? "";
          const loss = entry.loss ?? "";
          const acc = entry.eval_accuracy ?? "";
          const f1 = entry.f1 ?? entry.eval_f1 ?? "";

          row.innerHTML = `
            <td class="px-4 py-2">${{step}}</td>
            <td class="px-4 py-2">${{epoch}}</td>
            <td class="px-4 py-2">${{loss}}</td>
            <td class="px-4 py-2">${{acc}}</td>
            <td class="px-4 py-2">${{f1}}</td>
          `;
          tbody.appendChild(row);
        }});
      }})
      .catch(err => {{
        document.getElementById("log-body").innerHTML =
          '<tr><td colspan="5" class="px-4 py-2 text-red-500">Error loading log: ' + err + '</td></tr>';
      }});
  </script>
</body>
</html>
"""

with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(html.strip())

print("✅ Styled training report generated at docs/index.html")
