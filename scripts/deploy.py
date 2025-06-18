import json
from huggingface_hub import Repository, login
import os
from shutil import copytree
import shutil

# Load evaluation results
with open("results.json", "r") as f:
    results = json.load(f)

accuracy = results.get("accuracy", 0)
f1_score = results.get("f1_score", 0)

THRESHOLD = float(os.getenv("THRESHOLD_SCORE", "0.0000"))

if accuracy >= THRESHOLD or f1_score >= THRESHOLD:
    print(f"Scores are good (accuracy: {accuracy:.4f}, f1: {f1_score:.4f}). Starting deployment...")

    repo_id = "your-username/sentiment-distilbert"  # Change this!

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Please set your HF_TOKEN environment variable with your Hugging Face access token")

    login(token=hf_token)

    repo_dir = "."  # Use current folder (your existing local repo)
    repo = Repository(local_dir=repo_dir, use_auth_token=hf_token)

    dest_dir = os.path.join(repo_dir, "distilbert-sentiment")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    copytree("./models/distilbert-sentiment", dest_dir)

    repo.git_add()
    repo.git_commit("Deploying model after evaluation passed")
    repo.git_push()

    print("Model deployed successfully to Hugging Face Hub!")

else:
    print(f"Scores below threshold (accuracy: {accuracy:.4f}, f1: {f1_score:.4f}). Deployment skipped.")
