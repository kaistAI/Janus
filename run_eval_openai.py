import asyncio
import glob
import json
import os
import pickle
import random
import warnings
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from utils.data_utils import get_load_func
from utils.models.openai_utils import OpenAILLM
from utils.output_parser import parse_judgment_abs
from utils.prompts import (
    ABS_SYSTEM_PROMPT,
    ABS_USER_PROMPT_TEMPLATE,
    SAMPLING_PARAMS_OPENAI,
)

DEBUG = False


# Moddel inference (Use offline batching)
async def batch_completions_with_retries(
    model,
    inputs,
    batch_size,
    parse_output,
    max_retries=5,
):
    batched_outputs = []

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
        batch_outputs = await model.completions(batch_inputs, **SAMPLING_PARAMS_OPENAI)
        batched_outputs.extend(batch_outputs)

    # Identify failed instances and prepare for retries
    to_retry_inputs = []
    to_retry_indices = []
    for i, output in enumerate(batched_outputs):
        feedback, score = parse_output(output)
        if feedback is None:  # Parsing failed
            to_retry_inputs.append(inputs[i])
            to_retry_indices.append(i)

    # Retry logic with progress bar
    retries = 0
    while to_retry_inputs and retries < max_retries:
        retries += 1
        print(f"Retrying failed batches: Attempt {retries}/{max_retries}")
        retry_outputs = []
        for i in tqdm(
            range(0, len(to_retry_inputs), batch_size), desc=f"Retry Attempt {retries}"
        ):
            batch_inputs = to_retry_inputs[i : i + batch_size]
            batch_outputs = await model.completions(
                batch_inputs, **SAMPLING_PARAMS_OPENAI
            )

            assert len(batch_outputs) == len(batch_inputs)
            retry_outputs.extend(batch_outputs)

        new_to_retry_inputs = []
        new_to_retry_indices = []
        for idx, (retry_idx, output) in enumerate(zip(to_retry_indices, retry_outputs)):
            feedback, score = parse_output(output)
            if feedback is None:  # Still failing
                new_to_retry_inputs.append(to_retry_inputs[idx])
                new_to_retry_indices.append(to_retry_indices[idx])
            else:
                batched_outputs[retry_idx] = output  # Update with successful retry

        to_retry_inputs = new_to_retry_inputs
        to_retry_indices = new_to_retry_indices

    # Final aggregation and printing
    outputs_len = len(batched_outputs)
    print(f"Processed {outputs_len}/{total_len} instances.")

    if outputs_len < total_len:
        warnings.warn("Some instances failed to generate feedback.")
        warnings.warn("They will be written as None in the output file.")
        raise Exception(
            f"Failed to generate feedback for {total_len - outputs_len} instances."
        )

    feedbacks = []
    scores = []

    for output in tqdm(batched_outputs, desc="Finalizing"):
        feedback, score = parse_output(output)
        if feedback is not None:
            feedbacks.append(feedback)
            scores.append(score)
        else:
            # raise Exception(
            #     f"Parsing failed for output: {output}. Feedback: {feedback}, Score: {score}"
            # )
            feedbacks.append(None)
            scores.append(None)

    if DEBUG:
        print("Checking the results")
        print(*list(zip(feedbacks, scores))[:10])

    return feedbacks, scores


def prepare_inputs(
    records,
    user_key,
    answer_key,
    rubric_key,
    system_key=None,
):
    inputs = []
    preference_set_indices = [0]

    for record in records:
        response = record["response"]

        if system_key and system_key in record:
            instruction = f"{record[system_key]}\n{record[user_key]}"
        else:
            instruction = record[user_key]

        reference_answer = record[answer_key]

        rubrics = record[rubric_key]

        if isinstance(rubrics, list):
            for rubric in rubrics:
                rubric_str = json.dumps(rubric, indent=4)

                content = ABS_USER_PROMPT_TEMPLATE.format(
                    instruction=instruction,
                    response=response,
                    reference_answer=reference_answer,
                    score_rubric=rubric_str,
                ).strip()

                messages = [
                    {"role": "system", "content": ABS_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ]
                inputs.append(messages)
        else:
            content = ABS_USER_PROMPT_TEMPLATE.format(
                instruction=instruction,
                response=response,
                reference_answer=reference_answer,
                score_rubric=rubrics,
            ).strip()

            messages = [
                {"role": "system", "content": ABS_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            inputs.append(messages)
        preference_set_indices.append(len(inputs))

    random_inputs = random.sample(inputs, 3)
    width = 20

    for input_str in random_inputs:
        print("-" * width)
        print("Example inputs:")
        print(input_str)
    print("-" * width)
    return inputs, preference_set_indices


async def main(args):
    load_func = get_load_func(args.input_file)

    data_dict = {d["id"]: d for d in load_func(args.input_file)}
    data_list = list(data_dict.values())

    if args.response_dir and args.response_file:
        raise ValueError(
            "Only one of response_dir or response_file should be provided."
        )

    if args.response_dir:
        pattern = os.path.join(args.response_dir, "*_responses.json")
        response_files = glob.glob(pattern)
    elif args.response_file:
        response_files = [args.response_file]
    else:
        raise ValueError("Either response_dir or response_file must be provided.")

    model = OpenAILLM(args.model_name)

    for file_path in response_files:
        print(f"Loading file: {file_path}")
        response_model_name = file_path.split("/")[-1].replace("_responses.json", "")
        judgment_dict = data_dict.copy()

        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        for id, record in data.items():
            if id not in data_dict.keys():
                assert 0
            data_dict[id].update({"response": record["response"]})
        data_list = list(data_dict.values())
        inputs, preference_set_indices = prepare_inputs(
            data_list,
            args.user_key,
            args.answer_key,
            args.rubric_key,
            args.system_key,
        )

        output_dir = os.path.join(args.output_dir, "responses_gpt4_eval")
        output_file = Path(output_dir) / f"{response_model_name}_evaluation.json"
        print("Output file: ", str(output_file))

        if output_file.exists() and not args.force_rerun:
            print("Output file already exists. Run Finished.")
            continue

        output_file.parent.mkdir(parents=True, exist_ok=True)

        batch_size = 100

        # DEBUG: Debugging purposes
        if DEBUG:
            inputs = inputs[:10]
            data_list = data_list[:10]

        feedbacks, scores = await batch_completions_with_retries(
            model, inputs, batch_size, parse_judgment_abs
        )

        # assert len(feedbacks) == len(scores)
        # assert len(feedbacks) == len(data_list)

        avg_score = 0.0

        for idx, instance in enumerate(data_list):
            preference_set_idx_start = preference_set_indices[idx]
            preference_set_idx_end = preference_set_indices[idx + 1]
            feedbacks_set = feedbacks[preference_set_idx_start:preference_set_idx_end]
            scores_set = scores[preference_set_idx_start:preference_set_idx_end]
            judgment_dict[instance["id"]].update(
                {"feedback": feedbacks_set, "score": scores_set}
            )
            avg_score += sum(scores_set) / len(scores_set)

        avg_score /= len(data_list)

        with output_file.open("w") as file:
            file.write(json.dumps(judgment_dict, indent=4))

        print(f"Average score: {avg_score}")

        if DEBUG:
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4-0125-preview")
    parser.add_argument("--input_file", type=str)
    parser.add_argument(
        "--response_dir",
        type=str,
        default=None,
        help="Path to the response directory; every response file should be '{{model_name}}_responses.json'",
    )
    parser.add_argument(
        "--response_file",
        type=str,
        default=None,
        help="Path to the response file (json); response file should be '{{model_name}}_responses.json'",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, default=os.path.dirname(__file__)
    )
    parser.add_argument("--user_key", type=str, required=True, default="instruction")
    parser.add_argument("--answer_key", type=str, required=True, default="reference_answer")
    parser.add_argument("--rubric_key", type=str, default="rubric")
    parser.add_argument("--system_key", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args))
