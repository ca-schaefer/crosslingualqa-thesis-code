"""
Splits the corpus in possible-to-resolve questions and unanswerable questions
"""

import argparse
import json
import os
from os.path import join
import sys

from tqdm import tqdm

from data import load_xqa, save_xqa


def split_ids(data):
    answerable = []
    unanswerable = []
    for item in data.values():
        if item.check_gold_answer():
            answerable.append(item.id)
        else:
            unanswerable.append(item.id)
    return answerable, unanswerable


def filter_corpus(question_file, question_out_file, kept_ids, gold_file, gold_out_file):
    kept_questions = set()
    with open(question_file, encoding="utf-8") as fin:
        with open(question_out_file, "w", encoding="utf-8") as fout:
            for line in tqdm(fin):
                json_data = json.loads(line)
                current_output = []
                for item in json_data:
                    if item["id"][0] in kept_ids:
                        current_output.append(item)
                        kept_questions.add(item["question"])
                if current_output:
                    json.dump(current_output, fout, ensure_ascii=False)
                    fout.write("\n")

    # Gold data are not saved with id, filter with questions string
    with open(gold_file, encoding="utf-8") as fin:
        with open(gold_out_file, "w", encoding="utf-8") as fout:
            for line in fin:
                json_gold = json.loads(line)
                if json_gold["question"] in kept_questions:
                    json.dump(json_gold, fout, ensure_ascii=False)
                    fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a corpus into answerable and unanswerable questions")
    parser.add_argument("-c", "--corpus", help="Path to the corpus")
    parser.add_argument("-a", "--answerable", help="Path to the first output directory")
    parser.add_argument("-u", "--unanswerable", help="Path to the second output directory")
    parser.add_argument("-p", "--part", help="train, dev or test part of the corpus", default="dev",
                        choices=["train", "dev", "test"])

    args = parser.parse_args()

    corpus_part = args.part

    # load data
    question_data = join(args.corpus, f"{corpus_part}_doc.json")
    gold_data = join(args.corpus, f"{corpus_part}.txt")

    # prepare output directories
    answerable_dir = args.answerable
    unanswerable_dir = args.unanswerable
    try:
        os.makedirs(answerable_dir)
    except FileExistsError:
        print(f"Directory already exists: {answerable_dir}")

    try:
        os.makedirs(unanswerable_dir)
    except FileExistsError:
        print(f"Directory already exists{unanswerable_dir}")

    data = load_xqa(question_data, gold_data)
    print(f"Data loaded of size {len(data)}")

    answerable, unanswerable = split_ids(data)

    question_out_file = join(answerable_dir, f"{corpus_part}_doc.json")
    gold_out_file = join(answerable_dir, f"{corpus_part}.txt")
    filter_corpus(question_data, question_out_file, answerable, gold_data, gold_out_file)

    question_out_file = join(unanswerable_dir, f"{corpus_part}_doc.json")
    gold_out_file = join(unanswerable_dir, f"{corpus_part}.txt")
    filter_corpus(question_data, question_out_file, unanswerable, gold_data, gold_out_file)

