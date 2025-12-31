import os
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="WestlakeNLP/Research-14K",
    repo_type="dataset",
    local_dir="Research-14K",
    local_dir_use_symlinks=False
)
