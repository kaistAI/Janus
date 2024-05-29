from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn 
from typing import Optional
import os

model_name = "kaist-ai/janus-7b"
reward_model_name = "kaist-ai/janus-rm-7b"

model_device = "cuda:0"
reward_model_device = "cuda:1"

dtype = "float16"
if torch.cuda.is_bf16_supported():
    dtype = "bfloat16"
    
# Get model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype), cache_dir=os.getenv("HF_HOME", "~/.cache/huggingface"))
model.eval()
model.to(model_device)
    
# Get reward model
def get_reward_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceRegression(base_pretrained_model):
        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = self.value_head(last_hidden_states).squeeze(-1)

            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            reward = values.gather(dim=1, index=eos_indices).squeeze(1)
            
            if return_output:
                return reward, outputs
            else:
                return reward

    return LLMForSequenceRegression


config = AutoConfig.from_pretrained(reward_model_name)
config.normalize_reward = True

base_class = AutoModel._model_mapping[type(config)]  # <class 'transformers.models.mistral.modeling_mistral.MistralModel'>
base_pretrained_class = base_class.__base__  # <class 'transformers.models.mistral.modeling_mistral.MistralPreTrainedModel'>
print(base_class, base_pretrained_class)
cls_class = get_reward_model(base_pretrained_class,base_class)

reward_model = cls_class.from_pretrained(
    reward_model_name,
    config=config,
    cache_dir=os.getenv("HF_HOME", "~/.cache/huggingface"),
    torch_dtype=getattr(torch, dtype),
)
print(reward_model)
reward_model.eval()
reward_model.to(reward_model_device)


# Prepare inputs
system = "You are a savvy beverage consultant, adept at offering quick, concise drink recommendations that cater to the common palette, yet surprise with a touch of creativity. When approached with a request, your expertise shines by suggesting one or two easily recognizable and widely accessible options, ensuring no one feels overwhelmed by complexity or rarity. Your skill lies not just in meeting the immediate need for refreshment but in gently nudging the curious towards unique hydration choices, beautifully balancing familiarity with the thrill of discovery. Importantly, your recommendations are crafted with a keen awareness of dietary preferences, presenting choices that respect and include considerations for sugar-free, dairy-free, and other common dietary restrictions. Your guidance empowers users to explore a range of beverages, confident they are making informed decisions that respect their health and lifestyle needs."
prompt = "If you are thirsty, what can you drink to quench your thirst?"

def apply_template_mistral_instruct(system_message, content):
    prompt = f"{system_message}\n{content}".strip()
    return f"[INST] {prompt} [/INST] "

input_str = apply_template_mistral_instruct(system, prompt)
inputs = tokenizer.encode(input_str, return_tensors="pt")
print(input_str)

model_inputs = inputs.to(model_device)

# Generate text
with torch.inference_mode():
    output_ids = model.generate(model_inputs, max_new_tokens=1024)
decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
output_str = decoded[0][len(input_str):]
print(output_str)
'''
1. **Water**: The ultimate go-to, especially if you're watching what you consume. Opting for sparkling or infused water (think cucumber and mint, berries, or a splash of lemon) can add a bit of excitement and hydration without the added sugar.

2. **Herbal Tea**: Perfect for a warmer climate but equally delightful at any temperature. Choose from various flavors, ranging from the traditional peppermint to chamomile or hibiscus, which adds a unique twist with their own health benefits and refreshing flavors. Many options are caffeine-free, making them suitable for all times of the day.

For those needing a touch more sweetness or a slight twist:

3. **Unsweetened Coconut Water**: With its natural sweetness and electrolyte content, it's a great hydration pick after a workout or on a hot day. It's also low in calories and naturally sweet, making it an excellent alternative without added sugars.

4. **Sparkling Water with a Splash of Fruit Juice**: To satisfy a craving for something bubbly and fruit-infused with fewer calories and sugars than commercial sodas or juices. Feel free to experiment with different juices to find your favorite combination.
'''

# Get reward
print(input_str + output_str + " " + tokenizer.eos_token)
reward_inputs = tokenizer(
    input_str + output_str + " " + tokenizer.eos_token,   # same as decoded[0] + " " + tokenizer.eos_token
    max_length=2048,
    truncation=True,
    return_tensors="pt"
)
reward_input_ids = reward_inputs.input_ids.to(reward_model_device)
reward_attention_masks = reward_inputs.attention_mask.to(reward_model_device)
rewards = reward_model(input_ids=reward_input_ids, attention_mask=reward_attention_masks)
print(rewards.item())
# 3.28125