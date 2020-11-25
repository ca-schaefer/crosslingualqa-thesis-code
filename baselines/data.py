"""
Representation of the XQA corpus and related functions
"""

import argparse
import json
import logging
import os
from os.path import join
from typing import Dict, List


class BaselineData(object):

    def __init__(self, iid: int, question: str, documents: List, document_ids: List, gold=[]):
        self.id = iid
        self.question = question
        self.documents = documents
        self.document_ids = document_ids
        self.gold = gold

    def add_gold_answer(self, answer):
        self.gold = answer

    def __str__(self):
        return f"id {self.id}: {self.question} - {self.gold}"

    def check_gold_answer(self, n=None):
        """Returns true if the documents contains at least one of the gold answers.
        Args:
            n: number of n best documents that should be checked for the answer, None for all documents
        """
        if n is None:
            current_documents = self.documents
        elif n <= len(self.documents):
            current_documents = self.documents[:n]
        else:
            current_documents = self.documents
        for answer in self.gold:
            logging.debug(f"Current gold answer {answer}")
            for document in current_documents:
                if document.find(answer) >= 0:
                    return True
        return False

    def get_json(self):
        """Returns a json representation of the data"""
        json_item = {"id: ": self.id,
                     "question: ": self.question,
                     "documents: ": self.documents,
                     "document_ids: ": self.document_ids,
                     "gold answers: ": self.gold}
        return json_item

    def get_xqa_json(self):
        """Returns a json representation in the format of one line of XQA dev.json"""
        line = []
        for count, document in enumerate(self.documents):
            json_item = {"id": [self.id, count],
                         "question": self.question,
                         "document": document,
                         "document_id": self.document_ids[count]}
            line.append(json_item)
        return line

    def get_answer_json(self):
        """Returns a json representation in the format of one line of XQA dev.txt"""
        res = {"answers": self.gold,
               "question": self.question}
        return res


def load_xqa_wrapper(path: str, part: str):
    """Loads the correct files from a given path"""
    if part == "all":
        parts = ["dev", "test"]
        data = {}
        if os.path.exists(join(path, "train_doc.json")):
            parts.append("train")
        for part in parts:
            question_data = join(path, f"{part}_doc.json")
            gold_data = join(path, f"{parts}.txt")
            data = {**data, **load_xqa(question_data, gold_data)}
    else:
        question_data = join(path, f"{part}_doc.json")
        gold_data = join(path, f"{part}.txt")
        data = load_xqa(question_data, gold_data)
    logging.info(f"Data loaded of size {len(data)}")
    return data


def load_xqa(question_file, gold_file):
    """Reads development data and returns it as dictionary {question: BaselineData}"""
    data = {}
    question = None
    all_docs = []
    doc_ids = []
    q_id = 0
    with open(question_file) as f:
        for line in f:
            json_data = json.loads(line)
            for item in json_data:
                iid = item["id"]
                if iid[0] != q_id:
                    if question is not None:
                        data[question] = (BaselineData(q_id, question, all_docs, doc_ids))
                    all_docs = []
                    doc_ids = []
                    q_id = iid[0]
                question = item["question"]
                document = item["document"]
                all_docs.append(document)
                doc_ids.append(item["document_id"])
        data[question] = (BaselineData(q_id, question, all_docs, doc_ids))

    with open(gold_file) as f:
        for line in f:
            json_gold = json.loads(line)
            answers = json_gold["answers"]
            question = json_gold["question"]
            data[question].add_gold_answer(answers)
    return data


def save_xqa(data: Dict, question_file: str, gold_file: str):
    """Saves corpus data in the XQA json format"""
    with open(question_file, "w") as f_question:
        with open(gold_file, "w") as f_gold:
            for item in data.values():
                json.dump(item.get_xqa_json(), f_question, ensure_ascii=False)
                f_question.write("\n")
                json.dump(item.get_answer_json(), f_gold, ensure_ascii=False)
                f_gold.write("\n")


def save_questionless_xqa(data: Dict, question_file: str, gold_file: str):
    """Saves corpus data in the XQA json format but substitutes the questions with blanks"""
    with open(question_file, "w") as f_question:
        with open(gold_file, "w") as f_gold:
            for item in data.values():
                current_json = item.get_xqa_json()
                current_json["question"] = ""
                json.dump(current_json, f_question, ensure_ascii=False)
                f_question.write("\n")
                json.dump(item.get_answer_json(), f_gold, ensure_ascii=False)
                f_gold.write("\n")


def load_and_save_corpus():
    question_file = "<TODO>/data/de/dev_doc.json"  #  TODO: add path here and next lines
    gold_file = "<TODO>/data/de/dev.txt"
    new_question_file = "<TODO>/data/de/copy_dev_doc.json"
    new_gold_file = "<TODO>/data/de/copy_dev.txt"
    data = load_xqa(question_file, gold_file)
    save_xqa(data, new_question_file, new_gold_file)


def check_answers_corpus():
    parser = argparse.ArgumentParser(description="Check proportion of answerable questions")
    parser.add_argument("corpus", help="Path to the eval data")
    parser.add_argument("-n", "--n_best", help="answerable with n best documents", type=int)
    parser.add_argument("-p", "--part", choices=["train", "dev", "test", "all"])

    args = parser.parse_args()

    # load data
    data = load_xqa_wrapper(args.corpus, args.part)

    # check number of found answers
    not_found = 0
    for question, item in list(data.items()):
        print(question)
        if not item.check_gold_answer(args.n_best):
            print(item.gold, item.question)
            not_found += 1

    print(f"{not_found} of {len(data)} questions without answer: {not_found / len(data)} %")


if __name__ == "__main__":
    # load_and_save_corpus()
    check_answers_corpus()

