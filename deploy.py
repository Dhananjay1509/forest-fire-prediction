import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
username = os.getenv("HF_USERNAME")  # Add this to .env
space_name = os.getenv("HF_SPACE_NAME", "forest-fire-prediction")  # Add this to .env

if not token or not username:
    raise ValueError("HF_TOKEN and HF_USERNAME environment variables must be set!")

api = HfApi(token=token)
space_id = f"{username}/{space_name}"

try:
    api.create_repo(
        repo_id=space_name,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True
    )

    api.upload_folder(
        repo_id=space_id,
        folder_path=".",
        repo_type="space",
        ignore_patterns=["venv/*", ".env", ".git/*", "__pycache__/*", "*.pyc"]
    )

    print(f"✅ Successfully deployed to: https://huggingface.co/spaces/{space_id}")

except Exception as e:
    print(f"❌ Error during deployment: {str(e)}")

