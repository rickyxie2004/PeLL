from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="camel-ai/physics", repo_type="dataset", filename="physics.zip",
                local_dir="/home/xieruiqi/DeepLoRA/", local_dir_use_symlinks=False)