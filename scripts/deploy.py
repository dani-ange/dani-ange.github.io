import json
from huggingface_hub import Repository, login
import os
from shutil import copytree, rmtree
from google.colab import userdata
# --- Load Evaluation Scores ---
with open("results.json", "r") as f:
    results = json.load(f)

accuracy = results.get("accuracy", 0)
f1 = results.get("f1_score", 0)

# --- Threshold from environment (default to 0.8) ---
THRESHOLD = 0.0000 #float(os.getenv("THRESHOLD_SCORE", 0.8))

# --- Check if model should be deployed ---
if accuracy >= THRESHOLD and f1 >= THRESHOLD:
    print(f"✅ Scores are good (accuracy: {accuracy:.4f}, f1: {f1:.4f}). Starting deployment...")

    # --- Hugging Face setup ---
    hf_token = userdata.get("HF_TOKEN")
    repo_id = "danielle2003/sentiment-analysis"  # ⚠️ REPLACE with your actual repo ID

    if not hf_token:
        raise EnvironmentError("❌ Please set the HF_TOKEN environment variable.")

    login(token=hf_token)

    # --- Prepare local repo ---
    repo_dir = "./hf_repo"
    if os.path.exists(repo_dir):
        rmtree(repo_dir)

    repo = Repository(local_dir=repo_dir, clone_from=repo_id, use_auth_token=hf_token)

    # --- Copy model files ---
    dest = os.path.join(repo_dir, "distilbert-sentiment")
    if os.path.exists(dest):
        rmtree(dest)

    copytree("./models/distilbert-sentiment", dest)
    import subprocess

    subprocess.run(['git', 'config', '--global', 'user.email', 'danielle@example.com'], check=True)
    subprocess.run(['git', 'config', '--global', 'user.name', 'Danielle'], check=True)

    # --- Commit and push ---
    repo.git_add()
    repo.git_commit("🚀 Deploying model after successful evaluation")
    try:
        repo.git_push()
        print("✅ Model deployed successfully to Hugging Face Hub!")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")

else:
    print(f"⚠️ Scores too low (accuracy: {accuracy:.4f}, f1: {f1:.4f}). Deployment skipped.")
