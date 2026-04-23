import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#meta-llama/Meta-Llama-3-8B-Instruct
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
print("Device to run the model on: ", model.device)

text = "Who are you?"
#inputs = tokenizer(text, return_tensors="pt").to(model.device)