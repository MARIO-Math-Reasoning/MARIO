# !/usr/bin/env python
# -*-coding:utf-8 -*-


import argparse
import codecs
import json
import sys

from tqdm import tqdm

from math_evaluation.core.evaluations import is_equiv
from math_evaluation.core.preprocess import *
from math_evaluation.core.metamath_util import is_equiv as metamath_is_equiv

def parse_args():
    parser = argparse.ArgumentParser(description="String Match based Eval Math Result")
    parser.add_argument("--input-file", type=str, default=None, help="input file.")
    parser.add_argument("--label-preprocess", type=str, default="gsm8k_label",
                        help="label should be preprocessed by function [label-preprocess]")
    parser.add_argument("--predicted-preprocess", type=str, default="gsm8k_predict",
                        help="predicted should be preprocessed by function [predicted-preprocess]")
    parser.add_argument("--debug", action="store_true", help="debug mode.")
    args = parser.parse_args()
    return args


def get_preprocess_func(args):
    try:
        return  globals()[args.label_preprocess], globals()[args.predicted_preprocess]
    except:
        print(f"label_preprocess: {args.label_preprocess}")
        print(f"predicted_preprocess: {args.predicted_preprocess}")
        raise

def main():
    args = parse_args()
    print("args: {{{")
    for k, v in sorted(vars(args).items()):
        print(f"\t{k}: {v}")
    print("}}}\n")

    with open(args.input_file, 'r') as fin:
        data_list = fin.readlines()

    label_preprocess, predicted_preprocess = get_preprocess_func(args)

    right_count = 0
    total_count = 0
    for item_json_string in tqdm(data_list):
        item = json.loads(item_json_string)
        total_count += 1
        ref_str = item["label"]
        answer = item['predict']

        ref_answer = label_preprocess(ref_str)
        predicted_answer = predicted_preprocess(answer)

        whether_right = is_equiv(ref_answer, predicted_answer, verbose=args.debug)
        # whether_right = metamath_is_equiv(ref_answer, predicted_answer)

        if args.debug:
            print(f"ref_str: {ref_str} VS predicted_str: {predicted_answer} = {whether_right}")

        if whether_right:
            right_count += 1

    right_rate = (right_count / total_count) * 100
    print(f"Total Questions: {total_count}")
    print(f"Right Answer Count: {right_count}")
    print(f"Right Rate: {right_rate:.2f}%")


if __name__ == "__main__":
    main()
