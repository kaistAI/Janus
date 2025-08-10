from argparse import ArgumentParser
from utils.output_parser import parse_judgment_alphabet
import csv
import json

index_to_alphabet = {0: "A", 1: "B", 2: "C"}

def main(args):
    with open(args.output_csv_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["example_id", "question_index", "question_polarity", "context_condition", "category", "answer_gt", "answer_pred"])

        with open(args.response_json_file, "r") as f:
            data = json.load(f)

            for example_id, d in data.items():
                prediction = d["feedback"]
                answer_gt = index_to_alphabet[d["answer_label"]]
                _, result = parse_judgment_alphabet(prediction)
                if result is None:
                    print(f"{example_id}: Error parsing prediction {prediction}")
                csvwriter.writerow([example_id, d["question_index"], d["question_polarity"], d["context_condition"], d["category"], answer_gt, result])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--response_json_file", type=str)
    parser.add_argument("--output_csv_file", type=str)
    args = parser.parse_args()
    
    main(args)