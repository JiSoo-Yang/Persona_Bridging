# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="google/gemma-2b-it",
#     local_dir="./gemma-2b-it",
#     local_dir_use_symlinks=False,  # ✅ 요게 핵심!
# )

from huggingface_hub import snapshot_download
import os

HF_TOKEN = os.environ.get("HF_TOKEN")

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    local_dir="./Llama-3.2-3B-Instruct",
    token=HF_TOKEN,
)