"""
Convert MLQA into XQA format
"""

import argparse
import json
from typing import List
import sys
import os

from data import BaselineData, save_xqa


class MlqaData(BaselineData):

    def __init__(self, long_id: str, question: str, documents: List, document_ids: List, gold=[], gold_start=[]):
        super().__init__(int(long_id, 16), question, documents, document_ids, gold)
        self.id = long_id
        self.question = question
        self.documents = documents
        self.document_ids = document_ids
        self.gold = gold
        self.gold_start = gold_start

    def get_mlqa_json(self):
        # TODO: several questions can share the same context document
        json = {"context": self.documents[0],  # mlqa only has one context document per question
                "qas": [
                    {
                     "question": self.question,
                     "answers": [
                         {"text": self.gold[0],  # mlqa only has one answer per question
                          "answer_start": self.gold_start[0]}
                     ],
                     "id": self.id}
                ]}
        return json


def read_mlqa(filename):
    """Reads a corpus in MLQA format and returns BaselineData."""
    data = {}
    with open(filename) as fin:
        raw_data = json.load(fin)
        for item in raw_data["data"]:
            title = item["title"]
            for paragraph in item["paragraphs"]:
                text = paragraph["context"]
                for qa_pair in paragraph["qas"]:
                    question = qa_pair["question"]
                    iid = qa_pair["id"]
                    answers = []
                    answer_starts = []
                    for answer in qa_pair["answers"]:
                        answers.append(answer["text"])
                        answer_starts.append(answer["answer_start"])
                    data[question] = MlqaData(iid, question, [text], [title], answers, answer_starts)
    return data


def write_mlqa(data, filename):
    """Save a corpus in MLQA format."""
    all_json = {}
    version_number = 1.0
    all_json["version"] = version_number

    all_json["data"] = []
    current_title = None
    collected_paragraphs = []
    old_context = None
    for item in data:
        title = item.document_ids[0]
        if title != current_title and (current_title is not None):
            sub_json = {"title": current_title, "paragraphs": collected_paragraphs}
            all_json["data"].append(sub_json)

            collected_paragraphs = []
        question_json = item.get_mlqa_json()
        if question_json["context"] == old_context:
            collected_paragraphs[-1]["qas"].extend(question_json["qas"])
        else:
            collected_paragraphs.append(question_json)
            old_context = question_json["context"]
        current_title = title
    sub_json = {"title": current_title, "paragraphs": collected_paragraphs}
    all_json["data"].append(sub_json)

    with open(filename, "w") as fout:
        json.dump(all_json, fout)


def load_and_save_corpus():
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    data = read_mlqa(in_file)
    write_mlqa(data, out_file)


def main():
    parser = argparse.ArgumentParser("Converting MLQA corpus to XQA format")
    parser.add_argument("in_file", help="Path to MLQA corpus")
    parser.add_argument("out_path", help="Path where the result should be saved")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    path = args.out_path

    if not os.path.exists(path):
        os.makedirs(path)
    if args.test:
        json_filename = os.path.join(path, "test_doc.json")
        txt_filename = os.path.join(path, "test.txt")
    else:
        json_filename = os.path.join(path, "dev_doc.json")
        txt_filename = os.path.join(path, "dev.txt")
    data = read_mlqa(args.in_file)
    save_xqa(data, json_filename, txt_filename)


if __name__ == "__main__":
    main()

