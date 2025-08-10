import os
import random
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data_utils import get_load_func, get_save_func
from utils.models.vllm_utils import VLLM
from utils.output_parser import parse_judgment_abs

random.seed(42)

DEBUG = False

os.environ["CURL_CA_BUNDLE"] = ""

SAMPLING_PARAMS = {
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.9,
    "repetition_penalty": 1.03,
}


SCORE_RUBRIC_TEMPLATE = """
[{criteria}]
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
Score 4: {score4_description}
Score 5: {score5_description}
""".strip()

spec_rel_rubric = {
  "criteria": "Does the system message define specific preferences that are relevant to the user message and sufficiently steer the AI's response in a particular way?",
  "score1_description": "The system message fails to define any clear preferences or is entirely irrelevant to the user message, providing no guidance for the AI's response.",
  "score2_description": "The system message defines some preferences, but they are vague or only marginally relevant to the user message, offering minimal steering for the AI's response.",
  "score3_description": "The system message defines moderately specific preferences that are generally relevant to the user message, providing some steering for the AI's response, but lacking in certain key aspects.",
  "score4_description": "The system message defines clear, specific preferences that are relevant to the user message and provide good steering for the AI's response. Minor improvements could enhance the guidance.",
  "score5_description": "The system message defines highly specific preferences that are extremely relevant to the user message, providing comprehensive and precise steering for the AI's response in the intended direction."
}

coh_nat_rubric = {
  "criteria": "Are the explicit preferences in the system message realistic and written coherently?",
  "score1_description": "The preferences are unrealistic and incoherently written, with contradictory or illogical instructions that do not represent a plausible scenario.",
  "score2_description": "The preferences have significant issues with realism or coherence, containing awkward phrasing or implausible requirements that undermine their credibility.",
  "score3_description": "The preferences are somewhat realistic and coherently written, but may include occasional unrealistic elements or unclear phrasing that affect overall believability.",
  "score4_description": "The preferences are largely realistic and coherently written, representing a believable scenario with only minor issues in phrasing or logical flow.",
  "score5_description": "The preferences are highly realistic and perfectly coherent, reading as if written by/for a real user for a genuine task. They maintain a logical flow and represent a completely plausible real-world scenario."
}

system_eval_prompt = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
eval_prompt = """You need to rate the quality of a system message given a user message and a score rubric. In this case, the system message describes a role, situation, constraints, general instructions, etc. that in whole explicitly states a user's preference expecting a specific response to the user message. So, the system message steers the assistant's behavior to align with the user's intentions when responding to the user message; the system message itself is not a response to the user message. Evaluate the quality of a system message based on the criteria specified in the score rubric. Specifically,
1. Write a detailed feedback highlighting the strengths and/or weaknessess of the system message regarding the score criteria and description of each score.
2. Output a rating that is an integer in the range of 1 to 5, referring to the score rubric.
Please do not generate any other opening, closing, or unnecessary comments.

## Score rubric
{score_rubric}

## System message
{system_message}

## User message
{user_message}

## Output format
Output the following string, replacing the placeholders [...] with your words:

"Feedback: [...] [RESULT] [...]"

## Feedback: """

# Model inference (Use offline batching)
def batch_completions(
    model,
    inputs,
    batch_size,
    sampling_params: dict
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
        batch_outputs = model.completions(
            batch_inputs, **sampling_params, use_tqdm=True
        )

        for i, batch_output in enumerate(batch_outputs):  # len(batch_output) == N
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
    feedbacks = []
    scores = []
    for i, output in enumerate(batched_outputs):
        feedback, score = parse_judgment_abs(output)
        if feedback is not None:
            feedbacks.append(feedback)
            scores.append(score)
        else:
            # raise Exception(
            #     f"Parsing failed for output: {output}. Feedback: {feedback}, Score: {score}"
            # )
            feedbacks.append(output)
            scores.append(None)
            

    if DEBUG:
        print("Checking the results")
        print(batched_outputs[:10])

    return feedbacks, scores




def prepare_inputs(records, system_key, user_key, tokenizer, mode):
    inputs = []

    for record in records:
        system_message = record[system_key]
        user_message = record[user_key]
        
        content = eval_prompt.format(
            score_rubric=SCORE_RUBRIC_TEMPLATE.format(**spec_rel_rubric) if mode == "spec_rel" else SCORE_RUBRIC_TEMPLATE.format(**coh_nat_rubric),
            system_message=system_message,
            user_message=user_message,
        )
        
        messages = [
            {"role": "system", "content": system_eval_prompt},
            {"role": "user", "content": content},
        ]
        input_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
    # data_dict = {i: {"system": d["system"], "instruction": d["instruction"]} for i, d in enumerate(load_func(args.input_file))}
    # data_list = list(data_dict.values())

    data_list = []
    for d in load_func(args.input_file):
        if "id" not in d:
            d["id"] = len(data_list)
        data_list.append(d)
    data_dict = {d["id"]: {"system": d["system"], "instruction": d["instruction"]} for d in data_list}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # tokenizer.padding_side = "left"
    inputs = prepare_inputs(
        data_list, args.system_key, args.user_key, tokenizer, args.mode
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

    batch_size = 64

    # DEBUG: Debugging purposes
    if DEBUG:
        random_indices = random.sample(range(len(inputs)), 10)
        inputs = [inputs[i] for i in random_indices]
        data_list = [data_list[i] for i in random_indices]

    sampling_params = SAMPLING_PARAMS.copy()

    model = VLLM(args.model_name, num_gpus=args.num_gpus)
    feedbacks, scores = batch_completions(
        model, inputs, batch_size, sampling_params
    )

    
    avg_score = 0.0
    
    for idx, instance in enumerate(data_list):
        data_dict[instance["id"]].update(
            {"feedback": feedbacks[idx], "score": scores[idx]}
        )
        if scores[idx] is not None:
            avg_score += scores[idx]
    avg_score /= len(data_list)

    print(f"Saving to {str(output_file)}...")
    save_func = get_save_func(str(output_file))
    save_func(data_dict, str(output_file))
    
    print(f"Average score: {avg_score}")


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
    parser.add_argument("--mode", type=str, choices=["spec_rel", "coh_nat"])

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
