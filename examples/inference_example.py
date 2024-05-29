from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_name = "kaist-ai/janus-7b"
device = "cuda:0"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

dtype = "float16"
if torch.cuda.is_bf16_supported():
    dtype = "bfloat16"
    
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype), cache_dir=os.getenv("HF_HOME", "~/.cache/huggingface"))
model.eval()
model.to(device)

# Prepare inputs
system = "As a financial news headline writer with a flair for the dramatic, you have taken on the role of crafting compelling headlines about the integration of AI into the financial sector. Your expertise allows you to weave industry-specific terminology seamlessly into each headline, striking a balance between capturing attention and providing meaningful insights into the transformative benefits of AI in finance. With each headline, you focus on elucidating the key advantages AI brings to financial operations, making complex information accessible and immediately impactful. While your headlines are designed to engage and inform an audience of finance and technology professionals, you navigate the fine line of excitement and accuracy with care, ensuring that the promises made are grounded in reality, thus avoiding any form of sensationalism. Your mission is to distill the essence of AI's impact on finance into a single, powerful line that speaks volumes to the informed reader."
prompt = "Write a headline for an article about the benefits of using AI in the finance sector."

def apply_template_mistral_instruct(system_message, content):
    prompt = f"{system_message}\n{content}".strip()
    return f"[INST] {prompt} [/INST] "

input_str = apply_template_mistral_instruct(system, prompt)
inputs = tokenizer.encode(input_str, return_tensors="pt")
print(input_str)

model_inputs = inputs.to(device)

# Generate text
output_ids = model.generate(model_inputs, max_new_tokens=1024)
decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(decoded[0][len(input_str):])
# Revolutionary Trends: How AI Is Redefining Efficiency and Accuracy in the Financial Realm