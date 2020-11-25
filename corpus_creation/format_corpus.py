import html
import json
from os.path import join
import random
import re
import string

from tqdm import tqdm


def read_translated_json(filename: str):
    items = []
    with open(filename) as fin:
        for line in fin:
            items.append(json.loads(line.strip()))
    return items


def write_qa_txt(filename: str, items: list, translation: bool):
    with open(filename, "w", encoding="utf-8") as fout:
        for item in items:
            if translation:
                json_object = {"answers": item["translated_answers"], "question": item["translated_question"]}
            else:
                json_object = {"answers": item["answers"], "question": item["question"]}
            json.dump(json_object, fout, ensure_ascii=False)
            fout.write("\n")


def write_item_json(filename: str, items: list, translation: bool):
    with open(filename, "w", encoding="utf-8") as fout:
        for i, item in enumerate(items):
            if translation:
                json_object = {"id": [i, 0],
                               "question": item["translated_question"],
                               "document": item["target_text"],
                               "document_id": item["source_title"]}
            else:
                json_object = {"id": [i, 0],
                               "question": item["question"],
                               "document": item["target_text"],
                               "document_id": item["target_title"]}
            json.dump([json_object], fout, ensure_ascii=False)
            fout.write("\n")


def filter_without_document(path, part, translation: bool):
    # filename = join(path, f"{part}_translated.json")
    filename = join(path, f"{part}_single_doc_de_cleaned.json")
    data = read_translated_json(filename)
    filtered_data = []
    for item in data:
        if item["question"] != "<Query>" and item["target_text"] != "page does not exist" and not item["source_title"].endswith("(disambiguation)"):
            filtered_data.append(item)

    write_qa_txt(join(path, f"filtered_{part}_de.txt"), filtered_data, translation)
    write_item_json(join(path, f"filtered_{part}_single_doc_de.json"), filtered_data, translation)
    print("Original size ", len(data))
    print("Filtered size ", len(filtered_data))
    print("Discarded ", len(data) - len(filtered_data))


def plain_formatting(filename, part, translation):
    outfile_txt = f"/home/ca/Documents/Uni/Masterarbeit/crosslingual-qa-scripts/corpus_creation/{part}_de.txt"
    outfile_json = f"/home/ca/Documents/Uni/Masterarbeit/crosslingual-qa-scripts/corpus_creation/{part}_doc_de.json"
    data = read_translated_json(filename)
    write_qa_txt(outfile_txt, data, translation)
    write_item_json(outfile_json, data, translation)


def check_answer_in_title(path, part):
    txt_file = join(path, f"filtered_{part}_de.txt")
    json_file = join(path, f"filtered_{part}_single_doc_de.json")

    wrong_question_count = 0
    wrong_answer_count = 0
    total = 0
    next_line = True
    with open(txt_file) as ftxt:
        with open(json_file) as fjson:
            while next_line:
                txt_line = ftxt.readline()
                json_line = fjson.readline()
                if not txt_line:
                    break
                if not json_line:
                    print("txt longer")
                    break
                total += 1
                current_txt = json.loads(txt_line)
                current_json = json.loads(json_line)[0]
                if current_txt["question"] != current_json["question"]:
                    # print("wrong question")
                    # print(current_txt["question"])
                    # print(current_json["question"])
                    wrong_question_count += 1
                if current_json["document_id"] not in current_txt["answers"]:
                    # print("not in answers")
                    # print(current_json["document_id"])
                    # print(current_txt["answers"])
                    wrong_answer_count += 1
    print("Total: ", total)
    print("# wrong questions: ", wrong_question_count)
    print(wrong_question_count / total)
    print("# wrong answers: ", wrong_answer_count)
    print(wrong_answer_count / total)


def check_answer_in_doc(path, part):
    txt_file = join(path, f"filtered_{part}_de.txt")
    json_file = join(path, f"filtered_{part}_single_doc_de.json")

    wrong_question_count = 0
    wrong_answer_count = 0
    total = 0
    next_line = True
    with open(txt_file) as ftxt:
        with open(json_file) as fjson:
            while next_line:
                txt_line = ftxt.readline()
                json_line = fjson.readline()
                if not txt_line:
                    break
                if not json_line:
                    print("txt longer")
                    break
                total += 1
                current_txt = json.loads(txt_line)
                current_json = json.loads(json_line)[0]
                if current_txt["question"] != current_json["question"]:
                    # print("wrong question")
                    # print(current_txt["question"])
                    # print(current_json["question"])
                    wrong_question_count += 1
                text = current_json["document"]
                found = False
                for answer in current_txt["answers"]:
                    if answer in text:
                        found = True
                if not found:
                    print("not in answers")
                    print(current_json["document_id"])
                    print(current_txt["answers"])
                    wrong_answer_count += 1
    print("Total: ", total)
    print("# wrong questions: ", wrong_question_count)
    print(wrong_question_count / total)
    print("# wrong answers: ", wrong_answer_count)
    print(wrong_answer_count / total)


def strip_punctuation(text: str):
    # for char in string.punctuation:
    #     text = text.replace(char, "")
    # for char in string.whitespace:
    #     text = text.replace(char, "")
    text = html.unescape(text)
    text = re.sub(r"< i >", "", text)
    text = re.sub(r"< /i >", "", text)
    text = re.sub(r"& nbsp ;", "", text)
    text = re.sub(r"& amp ;", "", text)
    new_text = []
    for char in text:
        if char in string.ascii_letters:
            new_text.append(char)
    text = "".join(new_text)
    return text


def add_additional_docs(path, part):
    single_doc_file = join(path, f"filtered_{part}_single_doc_de.json")
    multi_doc_file = join(path, f"corpus_{part}_BM25_documents.json")
    out_file = join(path, f"{part}_doc_de.json")

    data = {}
    with open(single_doc_file) as fin:
        for line in tqdm(fin):
            current = json.loads(line.strip())
            if type(current) is list:
                question_key = strip_punctuation(current[0]["question"])
                data[question_key] = current
            else:
                question_key = strip_punctuation(current["question"])
                data[question_key] = [current]

    same_doc = 0
    with open(multi_doc_file) as fin:
        with open(out_file, "w") as fout:
            for line in tqdm(fin):
                current = json.loads(line.strip())
                question = current["question"]
                question_key = strip_punctuation(question)
                if question_key not in data:
                    # was filtered
                    print("Not in data ", question, question_key)
                    continue
                orig_doc = data[question_key][0]["document"]
                question = data[question_key][0]["question"]  # use the question version with less whitespace
                current_list = data[question_key].copy()
                for i, doc in enumerate(current["documents"]):
                    question_id = data[question_key][0]["id"][0]
                    new_item = {"id": [question_id, i],
                                "question": question,
                                "document": doc["text"],
                                "document_id": doc["id"]}
                    current_list.append(new_item)
                    if orig_doc == doc["text"]:
                        same_doc += 1
                if len(current_list) < 10:
                    print(question)
                json.dump(current_list[:10], fout, ensure_ascii=False)
                fout.write("\n")

    print("Total ", len(data))
    print("same docs ", same_doc)
    print("Fraction ", same_doc / len(data))


def add_article_title_as_answer(path, part):
    txt_file = join(path, f"filtered_{part}_de.txt")
    json_file = join(path, f"filtered_{part}_single_doc_de.json")
    out_file = join(path, f"filtered_{part}_de_extended.txt")

    answers = {}
    with open(json_file) as fin:
        for line in fin:
            current = json.loads(line.strip())
            answers[current[0]["question"]] = current[0]["document_id"]

    with open(txt_file) as fin:
        with open(out_file, "w") as fout:
            for line in fin:
                current = json.loads(line.strip())
                title_answer = answers[current["question"]]
                if title_answer not in current["answers"]:
                    current["answers"].append(title_answer)
                json.dump(current, fout, ensure_ascii=False)
                fout.write("\n")


def remove_first_paragraph(path, part):
    in_file = join(path, f"{part}_doc_de.json")
    out_file = join(path, f"{part}_minus_first_doc_de.json")
    with open(in_file) as fin:
        with open(out_file, "w") as fout:
            for line in tqdm(fin):
                current = json.loads(line.strip())
                for i in range(len(current)):
                    paragraphs = current[i]["document"].split("\n")
                    current[i]["document"] = "\n".join(paragraphs[1:])
                json.dump(current, fout, ensure_ascii=False)
                fout.write("\n")


def select_items(n: int, part):
    txt_file_in = f"created_corpus/filtered_{part}_de_extended.txt"
    json_file_in = f"created_corpus/filtered_{part}_minus_first_doc_de.json"
    txt_file_out = f"created_corpus/selection_{n}_{part}_de.txt"
    json_file_out = f"created_corpus/selection_{n}_{part}_doc_de.json"

    with open(txt_file_in) as fin:
        for i, _ in enumerate(fin):
            pass
        length = i + 1

    indices = random.sample(range(length), k=n)
    print(indices)
    print(len(indices))
    print("only unique ", len(indices) == len(set(indices)))
    questions = {}
    counter = 0
    with open(txt_file_in) as fin:
        with open(txt_file_out, "w") as fout:
            for i, line in enumerate(fin):
                if i in indices:
                    current = json.loads(line)
                    if current["question"] in questions:
                        print(current["question"])
                        print(i)
                        print(questions[current["question"]])
                        continue
                    fout.write(line)
                    questions[current["question"]] = (counter, i)
                    counter += 1
    print("questions", len(questions))
    lines = [None for _ in range(n)]
    print("lines ", len(lines))
    print("counter ", counter)
    with open(json_file_in) as fin:
        for line in fin:
            current = json.loads(line)
            question = current[0]["question"]
            if question in questions:
                idx = questions[question][0]
                if idx >= len(lines):
                    print(idx)
                lines[idx] = line

    with open(json_file_out, "w") as fout:
        for i, line in enumerate(lines):
            if line:
                fout.write(line)
            else:
                print(i)


if __name__ == "__main__":
    parts = ["dev", "train"]
    # parts = ["train"]
    path = "."
    for current_part in parts:
        print(current_part)
        # print("Filter without document")
        # filter_without_document(current_part)
        # print("Check answer in doc")
        # check_answer_in_doc(current_part)
        print("Add additional docs")
        add_additional_docs(path, current_part)
        print("Remove first paragraph")
        remove_first_paragraph(path, current_part)
        print("Add article title as answer")
        add_article_title_as_answer(path, current_part)
        print("select items")
        if current_part == "train":
            for sample_size in [500, 1000, 5000]:
                select_items(sample_size, current_part)
        # select_items(5000, current_part)
