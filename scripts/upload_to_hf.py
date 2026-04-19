import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_to_hf(repo_id: str, token: str, data_dir: str = "data"):
    """Uploads the local data directory to a Hugging Face Dataset repository."""
    api = HfApi()
    
    print(f"🚀 Creating repository '{repo_id}' if it doesn't exist...")
    try:
        create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating repo: {e}")
        return

    # Files to upload
    files_to_upload = []
    
    # 1. Processed Parquet
    parquet_path = os.path.join(data_dir, "papers_processed.parquet")
    if os.path.exists(parquet_path):
        files_to_upload.append((parquet_path, "papers_processed.parquet"))
    
    # 2. Index files
    index_dir = os.path.join(data_dir, "index")
    if os.path.exists(index_dir):
        for f in os.listdir(index_dir):
            if f.endswith((".index", ".pkl", ".json")):
                files_to_upload.append(
                    (os.path.join(index_dir, f), f"index/{f}")
                )

    if not files_to_upload:
        print("⚠️ No data files found in 'data/' directory. Build your index first!")
        return

    print(f"📦 Found {len(files_to_upload)} files to upload. Starting upload...")
    
    for local_path, path_in_repo in files_to_upload:
        print(f"⬆️ Uploading {path_in_repo}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    print(f"\n✅ All files uploaded successfully to https://huggingface.co/datasets/{repo_id}")
    print("\nNext step: Add these to your Streamlit Secrets:")
    print(f'HF_REPO_ID = "{repo_id}"')
    print(f'HF_TOKEN = "your_huggingface_token_here"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload index data to Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="Hugging Face repo ID (e.g., 'username/repo-name')")
    parser.add_argument("--token", required=True, help="Hugging Face Write Token")
    
    args = parser.parse_args()
    upload_to_hf(args.repo, args.token)
