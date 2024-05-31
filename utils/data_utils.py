import csv
import json
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

log_root = "./"


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def get_load_func(file):
    def load_json(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.values()

        for d in tqdm(data):
            yield d

    def load_jsonl(file_path):
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def load_csv(file_path):
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for line in reader:
                yield line
                
    def load_hf_dataset(file_path, split="train"):
        dataset = load_dataset(file_path, split=split)
        for d in dataset:
            yield d

    if file.endswith(".json"):
        load_func = load_json
    elif file.endswith(".jsonl"):
        load_func = load_jsonl
    elif file.endswith(".csv"):
        load_func = load_csv
    else:
        load_func = load_hf_dataset

    return load_func


def get_save_func(file):
    def save_json(data, file):
        with open(file, "w") as f:
            json.dump(data, f, indent=4)

    def save_jsonl(data, file):
        with open(file, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

    if file.endswith(".json"):
        save_func = save_json
    elif file.endswith(".jsonl"):
        save_func = save_jsonl
    else:
        raise ValueError(f"Invalid output file format {file}")

    return save_func


def set_logger(input_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    if input_file:
        # Create a file handler
        log_dir = os.path.join(
            log_root, os.path.splitext(os.path.basename(__file__))[0]
        )
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, os.path.basename(input_file).replace(".json", ".log")
        )
        print(f"Logging to {log_file}")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)

        # Create a logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger
