import os
import random
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data_utils import get_load_func, get_save_func, zero_pad_sequences
from utils.models.reward_model import get_llm_for_sequence_regression
from utils.models.vllm_utils import VLLM
from utils.prompts import SAMPLING_PARAMS

random.seed(42)

DEBUG = True

os.environ["CURL_CA_BUNDLE"] = ""


class CandidateCompletions(Dataset):
    def __init__(self, prompt, batch_output, tokenizer, max_length):
        super().__init__()
        self.prompt = prompt
        self.responses = batch_output
        self.tokenizer = tokenizer
        self.max_length = max_length  # input + output

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        response = self.responses[idx]
        input_tokens = self.tokenizer(
            self.prompt + response + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        info = {"input": self.prompt, "output": response}
        # to avoid EOS_token truncation
        input_tokens["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_tokens["attention_mask"][0][-1] = True
        return input_tokens["input_ids"], input_tokens["attention_mask"], info

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for input_id, attention_mask, info in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "left", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "left")
        return input_ids, attention_masks, infos


def get_best_response(completions_dataloader, reward_model):
    best_reward = -float("inf")
    best_response = None
    rewards = []
    with torch.no_grad():
        for input_ids, attention_masks, info in tqdm(
            completions_dataloader, desc="Reward Computation"
        ):
            input_ids = input_ids.squeeze(1).to(reward_model.device)
            attention_masks = attention_masks.squeeze(1).to(reward_model.device)
            rewards = reward_model(input_ids, attention_masks)
            for prompt, output, reward in zip(info["input"], info["output"], rewards):
                if reward.item() > best_reward:
                    best_reward = reward.item()
                    best_response = output
                rewards.append(reward.item())
    return best_response, best_reward, rewards


# Model inference (Use offline batching)
def batch_completions(
    model,
    inputs,
    batch_size,
    sampling_params: dict,
    reward_model: None,
    tokenizer: None,
):
    batched_outputs = []
    best_rewards = []

    # Adjust batch size to fit the number of inputs
    # VLLM supports adaptive batch size already
    total_batches = len(inputs) // batch_size + (
        1 if len(inputs) % batch_size > 0 else 0
    )
    total_len = len(inputs)

    n = sampling_params["n"]

    # Process initial batches with progress bar
    print("Processing initial batches...")
    for i in tqdm(
        range(0, len(inputs), batch_size), total=total_batches, desc="Initial Batches"
    ):
        batch_inputs = inputs[i : i + batch_size]
        batch_outputs = model.completions(
            batch_inputs, **sampling_params, use_tqdm=True
        )

        # Best-of-N sampling
        for i, batch_output in enumerate(batch_outputs):  # len(batch_output) == N
            if n > 1 and reward_model is not None:
                prompt = batch_inputs[i]
                # batchify reward computation
                candidate_completions = CandidateCompletions(
                    prompt,
                    batch_output,
                    tokenizer,
                    max_length=sampling_params["max_tokens"] * 2,
                )
                completions_dataloader = DataLoader(
                    candidate_completions,
                    batch_size=8,
                    drop_last=False,
                    collate_fn=candidate_completions.collate_fn,
                )
                best_response, best_reward, batch_rewards = get_best_response(
                    completions_dataloader, reward_model
                )
                best_rewards.append(best_reward)
                batched_outputs.append(best_response)

                if DEBUG:
                    print("Prompt:", prompt)
                    for response, reward in zip(batch_output, batch_rewards):
                        print(f"Response: {response} Score: {reward:.2f}")
            else:
                batched_outputs.append(batch_output[0])

    # Final aggregation and printing
    outputs_len = len(batched_outputs)
    print(f"Processed {outputs_len}/{total_len} instances.")

    if outputs_len < total_len:
        warnings.warn("Some instances failed.")
        warnings.warn("They will be written as None in the output file.")
        raise Exception(
            f"Failed to generate feedback for {total_len - outputs_len} instances."
        )

    for i, output in enumerate(batched_outputs):
        if output == "":
            print("Empty output")
            batched_outputs[i] = None

    if DEBUG:
        print("Checking the results")
        print(batched_outputs[:10])

    return batched_outputs, best_rewards


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
    # LLaMA-2 default chat template uses <<SYS>> and <</SYS>> for system messages
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama_fast.py#L36C1-L37C45


def apply_template_mistral_instruct(system_message, content):
    prompt = f"{system_message}\n{content}".strip()
    return f"[INST] {prompt} [/INST] "


def prepare_inputs(records, system_key, user_key, model_name: str, tokenizer):
    inputs = []

    for record in records:
        system_message = record[system_key]
        user_message = record[user_key]

        input_str = (
            apply_template_mistral_instruct(system_message, user_message)
            if "mistral" in model_name.lower() or "janus" in model_name.lower()
            else apply_template_chat(system_message, user_message, tokenizer)
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
    load_func = get_load_func(args.input_file)

    data_list = []
    for d in load_func(args.input_file):
        if "id" not in d:
            d["id"] = len(data_list)
        data_list.append(d)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    inputs = prepare_inputs(
        data_list, args.system_key, args.user_key, args.model_name, tokenizer
    )

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

    batch_size = 50

    # DEBUG: Debugging purposes
    if DEBUG:
        random_indices = random.sample(range(len(inputs)), 10)
        inputs = [inputs[i] for i in random_indices]
        data_list = [data_list[i] for i in random_indices]

    sampling_params = SAMPLING_PARAMS.copy()
    sampling_params.update(
        {
            "n": args.n,  # number of output sequences that are returned
        }
    )

    reward_model = None
    if args.n > 1 and args.reward_model_name:
        device = torch.device(f"cuda:{args.reward_model_device_num}")
        reward_model = get_llm_for_sequence_regression(
            args.reward_model_name,
            normalize_reward=True,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
        )
        reward_model.to(device)
        reward_model.eval()  # no training

    model = VLLM(args.model_name, num_gpus=args.num_gpus)
    outputs, rewards = batch_completions(
        model, inputs, batch_size, sampling_params, reward_model, tokenizer
    )

    assert len(outputs) == len(data_list)

    response_dict = {}
    if rewards:
        for instance, output, reward in zip(data_list, outputs, rewards):
            response_dict[instance["id"]] = {"response": output, "reward": reward}
    else:
        for instance, output in zip(data_list, outputs):
            response_dict[instance["id"]] = {"response": output}

    print(f"Saving to {str(output_file)}...")
    save_func = get_save_func(str(output_file))
    save_func(response_dict, str(output_file))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, default="responses/")
    parser.add_argument("--system_key", type=str, default="system")
    parser.add_argument("--user_key", type=str, default="prompt")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--force_rerun", action="store_true")

    # Best-of-N sampling
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of output sequences that are returned from the prompt.",
    )
    parser.add_argument("--reward_model_name", type=str)
    parser.add_argument("--reward_model_device_num", type=int, default=1)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    main(args)
