import os
import random
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.data_utils import get_load_func, get_save_func
from utils.prompts_bias import multiple_choice_system_prompts, multiple_choice_prompt
random.seed(42)

DEBUG = False

os.environ["CURL_CA_BUNDLE"] = ""

def token_to_extract(tokenizer):
    tokens_to_extract = {}
    for i in range(0x41, 0x44):  # A to C
        hex_value = hex(i).upper().replace('X', 'x')
        tokens = [f"<{hex_value}>", f"▁{chr(i)}", chr(i)]
        tokens_to_extract[chr(i)] = tokenizer.convert_tokens_to_ids(tokens)
    return tokens_to_extract


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # greedy decoding by default
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = model.config.eos_token_id

def get_model_response(
    prompts,
    model,
    tokenizer,
    max_tokens=30,
    batch_size=1
):
    result = []
    try:
        for idx, out in enumerate(tqdm(pipe(prompts, batch_size=batch_size, max_new_tokens=max_tokens), total=len(prompts))):
            raw = out[0]['generated_text']
            result.append(raw.split(prompts[idx])[-1].strip())
    except Exception as e:
        print(e)
    return result



# Model inference (Use offline batching)
def batch_completions(
    model,
    inputs,
    batch_size,
):
    batched_outputs = []
    
    # Adjust batch size to fit the number of inputs
    # VLLM supports adaptive batch size already
    total_batches = len(inputs) // batch_size + (
        1 if len(inputs) % batch_size > 0 else 0
    )
    total_len = len(inputs)

    # Process initial batches with progress bar
    print("Processing initial batches...")
    for i in tqdm(
        range(0, len(inputs), batch_size), total=total_batches, desc="Initial Batches"
    ):
        batch_inputs = inputs[i : i + batch_size]
        batch_outputs = get_model_response(
            batch_inputs,
            model,
            batch_size=batch_size
        )
        batched_outputs.extend(batch_outputs)

    # Final aggregation and printing
    outputs_len = len(batched_outputs)
    print(f"Processed {outputs_len}/{total_len} instances.")

    if DEBUG:
        print("Checking the results")
        print(batched_outputs[:10])

    return batched_outputs

def apply_template_chat(system_message, content, tokenizer):
    if tokenizer.chat_template and "system" not in tokenizer.chat_template:
        messages = [
            {"role": "user", "content": system_message + "\n" + content},
        ]
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": content},
        ]
    return (
        tokenizer.apply_chat_template(  # automatically format to default chat template
            messages, tokenize=False, add_generation_prompt=True
        )
    )

def apply_template_mistral_instruct(system_message, content):
    prompt = f"{system_message}\n{content}".strip()
    return f"[INST] {prompt} [/INST] "

def prepare_inputs(records, model_name, tokenizer):
    inputs = []

    for record in records:
        system_multichoice_prompt = random.choice(multiple_choice_system_prompts)
        
        a = record[f'ans{record["permuted_labels"][0]}']
        b = record[f'ans{record["permuted_labels"][1]}']
        c = record[f'ans{record["permuted_labels"][2]}']
        content = multiple_choice_prompt.format(
            context=record["context"],
            question=record["question"],
            a=a,
            b=b,
            c=c
        )
        input_str = (
            apply_template_mistral_instruct(system_multichoice_prompt, content)
            if "mistral" in model_name.lower() or "janus" in model_name.lower()
            else apply_template_chat(system_multichoice_prompt, content, tokenizer)
        )
        inputs.append(input_str)

    random_inputs = random.sample(inputs, 3)
    width = 20

    for input_str in random_inputs:
        print("-" * width)
        print("Example inputs:")
        print(input_str)
    print("-" * width)
    return inputs


def main(args):
    load_func = get_load_func(args.input_dataset)
    data_dict = {}
    labels = list(range(3))
    for d in load_func(args.input_dataset, split="test"):
        permuted_labels = labels.copy()
        random.shuffle(permuted_labels)
        answer_label = permuted_labels.index(d["label"])
        data_dict[d["example_id"]] = d | {
            "permuted_labels": permuted_labels,
            "answer_label": answer_label
        }
    data_list = list(data_dict.values())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    model = load_model(args.model_name)
    inputs = prepare_inputs(data_list, args.model_name, tokenizer)

    suffix = ""
    if args.suffix:
        suffix += "_" + args.suffix

    output_file = (
        Path(args.output_dir)
        / f"{args.model_name.split('/')[-1]}_responses{suffix}.json"
    )
    print(f"Output file: {str(output_file)}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not args.force_rerun:
        print("Output file already exists. Run Finished.")
        return

    batch_size = 64

    # DEBUG: Debugging purposes
    if DEBUG:
        random_indices = random.sample(range(len(inputs)), 10)
        inputs = [inputs[i] for i in random_indices]
        data_list = [data_list[i] for i in random_indices]

    outputs = batch_completions(
        model, inputs, batch_size
    )

    for idx, instance in enumerate(data_list):
        data_dict[instance["example_id"]].update(
            {"prediction": outputs[idx]}
        )

    print(f"Saving to {str(output_file)}...")
    save_func = get_save_func(str(output_file))
    save_func(data_dict, str(output_file))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--input_dataset", type=str, default="oskarvanderwal/bbq")
    parser.add_argument("--output_dir", type=str, required=True, default="responses/")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    main(args)
