import json
import os
import random
from argparse import ArgumentParser

from openai import OpenAI

from utils.data_utils import get_load_func, get_save_func
from utils.prompts import (
    ABS_SYSTEM_PROMPT,
    ABS_USER_PROMPT_TEMPLATE,
    SAMPLING_PARAMS_OPENAI,
)


class BatchClient:
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def create_batch(self, input_file, description=None):
        batch_input_file = self.client.files.create(
            file=open(input_file, "rb"), purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        batch_obj = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description},
        )
        return batch_obj

    def check_batch(self, batch_input_file_id):
        batch_obj = self.client.batches.retrieve(batch_input_file_id)
        return batch_obj

    def retrieve_batch(self, batch_input_file_id):
        content = self.client.files.content(batch_input_file_id)
        return content

    def cancel_batch(self, batch_input_file_id):
        self.client.batches.cancel(batch_input_file_id)

    def list_batches(self):
        return self.client.batches.list()


def prepare_inputs(
    records,
    eval_model_name: str,
    system_key=None,
    user_key="user_prompt",
    answer_key="reference_answer",
    rubric_key="rubric",
):
    inputs = []

    for record in records:
        id = record["id"]
        response = record["response"]

        if system_key and system_key in record:
            instruction = f"{record[system_key]}\n{record[user_key]}"
        else:
            instruction = record[user_key]

        reference_answer = record[answer_key]

        rubrics = record[rubric_key]

        for i, rubric in enumerate(rubrics):
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
            inputs.append(
                {
                    "custom_id": f"{id}_{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": eval_model_name,
                        "messages": messages,
                        **SAMPLING_PARAMS_OPENAI,
                    },
                }
            )

    random_inputs = random.sample(inputs, 3)
    width = 20

    for input_str in random_inputs:
        print("-" * width)
        print("Example inputs:")
        print(input_str)
    print("-" * width)
    return inputs


def prepare_input_file(input_file, response_file, eval_model_name):
    load_func_i = get_load_func(input_file)

    data_dict = {d["id"]: d for d in load_func_i(input_file)}
    data_list = list(data_dict.values())

    with open(response_file, "r") as f:
        responses = json.load(f)

    for id, response_record in responses.items():
        if id not in data_dict.keys():
            assert 0
        data_dict[id].update({"response": response_record["response"]})
    data_list = list(data_dict.values())

    inputs = prepare_inputs(data_list, eval_model_name)

    batch_eval_input_file = response_file.replace(
        "_responses.json", "_responses_batch_eval_input.jsonl"
    )
    save_func = get_save_func(batch_eval_input_file)
    save_func(inputs, batch_eval_input_file)

    return batch_eval_input_file


def main(args):
    client = BatchClient()
    if args.mode == "create":
        batch_eval_input_file = prepare_input_file(
            args.input_file, args.response_file, args.model
        )
        batch_obj = client.create_batch(batch_eval_input_file, args.description)
        print(batch_obj)
    elif args.mode == "check":
        batch_obj = client.check_batch(args.batch_id)
        print(batch_obj)
    elif args.mode == "retrieve":
        content = client.retrieve_batch(args.batch_output_file_id)
        with open(args.output_file, "w") as f:
            for line in content.iter_lines():
                f.write(line + "\n")
    elif args.mode == "cancel":
        client.cancel_batch(args.batch_id)
    elif args.mode == "list":
        batches = client.list_batches()
        print(batches)
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-i", "--input_file", type=str, help="MPA test data")
    parser.add_argument("-r", "--response_file", type=str)
    parser.add_argument("-d", "--description", type=str)
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output .jsonl file to write the retrieved content",
    )
    parser.add_argument("-b", "--batch_id", type=str)
    parser.add_argument("--batch_output_file_id", type=str)
    parser.add_argument(
        "--mode", type=str, choices=["create", "check", "retrieve", "cancel", "list"]
    )
    args = parser.parse_args()

    main(args)
