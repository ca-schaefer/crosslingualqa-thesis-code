import gzip
import json

from data import BaselineData


def read_tydi(in_file, language=None):
    data = {}
    with gzip.GzipFile(in_file, "r") as fin:
        for line in fin:
            raw_data = json.loads(line)
            text = raw_data["document_plaintext"]
            iid = raw_data["example_id"]
            question = raw_data["question_text"]
            document_id = raw_data["document_title"]
            gold = []
            for annotation in raw_data["annotations"]:
                start = annotation["minimal_answer"]["plaintext_start_byte"]
                end = annotation["minimal_answer"]["plaintext_end_byte"]
                answer_text = "NULL"
                if start >= 0 and end >= 0:
                    try:
                        answer_text = bytes(text, "utf-8")[start:end].decode()
                    except UnicodeDecodeError:
                        continue
                gold.append(answer_text)
            if language is None or language == raw_data["language"]:
                data[question] = BaselineData(iid, question, [text], [document_id], gold)
    return data

