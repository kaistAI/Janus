import random
import warnings
import asyncio

from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from utils.data_utils import get_load_func, get_save_func
from utils.models.openai_utils import OpenAILLM
from utils.prompts import SAMPLING_PARAMS_OPENAI

DEBUG = False


async def batch_completions(
    model,
    inputs,
    batch_size,
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

    # Final aggregation and printing
    outputs_len = len(batched_outputs)
    print(f"Processed {outputs_len}/{total_len} instances.")

    if outputs_len < total_len:
        warnings.warn("Some instances failed to generate feedback.")
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

    return batched_outputs


def prepare_inputs(records, system_key, user_key):
    inputs = []

    for record in records:
        user_message = record[user_key]
        if system_key and system_key in record:
            system_message = record[system_key]
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        else:
            messages = [
                {"role": "user", "content": user_message},
            ]

        inputs.append(messages)

    random_inputs = random.sample(inputs, 3)
    width = 20

    for input_str in random_inputs:
        print("-" * width)
        print("Example inputs:")
        print(input_str)
    print("-" * width)
    return inputs


async def main(args):
    load_func = get_load_func(args.input_file)

    data_list = []
    for d in load_func(args.input_file):
        if "id" not in d:
            d["id"] = len(data_list)
        data_list.append(d)

    model = OpenAILLM(args.model_name)

    inputs = prepare_inputs(data_list, args.system_key, args.user_key)

    output_file = (
        Path(args.output_dir) / f"{args.model_name.split('/')[-1]}_responses.json"
    )
    print(f"Output file: {str(output_file)}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not args.force_rerun:
        print("Output file already exists. Run Finished.")
        return

    batch_size = 100

    # DEBUG: Debugging purposes
    if DEBUG:
        inputs = inputs[:10]
        data_list = data_list[:10]

    outputs = await batch_completions(model, inputs, batch_size)

    assert len(outputs) == len(data_list)

    response_dict = {}
    for instance, output in zip(data_list, outputs):
        response_dict[instance["id"]] = {"response": output}

    print(f"Saving to {str(output_file)}...")
    save_func = get_save_func(str(output_file))
    save_func(response_dict, str(output_file))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, default="responses/")
    parser.add_argument("--system_key", type=str, default=None)
    parser.add_argument("--user_key", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args))
