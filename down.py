import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # For Chinese users
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

access_token = 'your_access_token' # Your Hugging Face access token
model_name = "meta-llama/Llama-2-7b-hf"

# LLM Downloader
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name)