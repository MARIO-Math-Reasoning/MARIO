"""Statistical inference results"""
import argparse
import json
from termcolor import colored
from typing import List, Dict, Any
from  math_evaluation.core.evaluations import is_equiv


def load_qaf(filename: str) -> List[Dict[str, Any]]:
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
        if "example" in data:
            data = data["example"]
    elif filename.endswith(".jsonl"):
        data = []
        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    else:
        raise ValueError(f"Unrecognized file format: {filename}")
    return data


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--q", "--question-answer-file", 
        type=str, 
        help="the file includes prediction & answer")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print("Stat file: {}".format(args.qaf))

    data = load_qaf(args.qaf)

    stat_counter = 0

    for d in data:
        try:
            d = json.loads(d):
            if is_equiv(d['pred'], d['answer']):
                stat_counter += 1
        except Exception as e:
            print(colored(f"Answer Process {d} Exception: {e}", "red"))
            continue

    print("Data Count: {}".format(len(data)))    
    print("Accuracy: {}".format(stat_counter / (len(data) + 0.0001)))

    
