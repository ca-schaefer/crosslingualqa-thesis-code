import json

from data import BaselineData


def read_xquad(in_file):
    data = {}
    with open(in_file) as fin:
        raw_data = json.load(fin)
        for parts in raw_data["data"]:
            for paragraph in parts["paragraphs"]:
                text = paragraph["context"]
                document_id = parts["title"]
                for qa_pair in paragraph["qas"]:
                    question = qa_pair["question"]
                    iid = qa_pair["id"]
                    answers = []
                    for answer in qa_pair["answers"]:
                        answers.append(answer["text"])
                    data[question] = BaselineData(iid, question, [text], [document_id], answers)
    return data


def read_xquad_context(in_file):
    data = {}
    text_collection = {}
    with open(in_file) as fin:
        raw_data = json.load(fin)
        for document in raw_data["data"]:
            texts = []
            document_id = document["title"]
            for paragraph in document["paragraphs"]:
                texts.append(paragraph["context"])
                for qa_pair in paragraph["qas"]:
                    question = qa_pair["question"]
                    iid = qa_pair["id"]
                    answers = []
                    for answer in qa_pair["answers"]:
                        answers.append(answer["text"])
                    data[question] = BaselineData(iid, question, [], [document_id], answers)
            text_collection[document_id] = "\n".join(texts)

    # Add texts
    for question in data:
        document_id = data[question].document_ids[0]
        text = text_collection[document_id]
        data[question].documents = [text]

    return data

