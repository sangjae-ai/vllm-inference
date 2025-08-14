import os
from huggingface_hub import snapshot_download


model_id = "google/gemma-3-4b-it"
local_model_path = "./gemma-3-4b-it"


snapshot_download(
    repo_id=model_id,
    local_dir=local_model_path,
    local_dir_use_symlinks=False
)
