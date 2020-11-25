import json
import os
from os.path import join


def remove_questions(in_path, out_path, substitute: bool):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    for part in ["train", "dev", "test"]:
        in_file = join(in_path, f"{part}.txt")
        if not os.path.exists(in_file):
            print("skipped part ", part, in_file)
            continue
        out_file = join(out_path, f"{part}.txt")
        with open(in_file) as fin:
            with open(out_file, "w") as fout:
                for line in fin:
                    current = json.loads(line)
                    if substitute:
                        current["question"] = " ".join("dummy" for _ in current["question"].split())
                    else:
                        current["question"] = ""
                    json.dump(current, fout, ensure_ascii=False)
                    fout.write("\n")

        in_file = join(in_path, f"{part}_doc.json")
        out_file = join(out_path, f"{part}_doc.json")
        with open(in_file) as fin:
            with open(out_file, "w") as fout:
                for line in fin:
                    current = json.loads(line)
                    for i, item in enumerate(current):
                        if substitute:
                            current[i]["question"] = " ".join("dummy" for _ in current[i]["question"].split())
                        else:
                            current[i]["question"] = ""
                    json.dump(current, fout, ensure_ascii=False)
                    fout.write("\n")


if __name__ == "__main__":
    original_path = "/home/ca/Documents/Uni/Masterarbeit/data/XQA_original/en"
    new_path = "/home/ca/Documents/Uni/Masterarbeit/data/XQA_no_question_plain/en"
    remove_questions(original_path, new_path, False)

