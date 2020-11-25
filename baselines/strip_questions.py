import argparse
import json
from os.path import join


DUMMY_QUESTION = "xxx xxx xxx xxx xxx"


def strip_question(infile, outfile):
    with open(infile) as fin:
        with open(outfile, "w") as fout:
            for line in fin:
                current = json.loads(line)
                current["question"] = DUMMY_QUESTION
                json.dump(current, fout, ensure_ascii=False)
                fout.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-l", "--language", default="en")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.input_path
    language = args.language

    parts = ["dev", "test"]
    if language == "en":
        parts.append("train")

    for part in parts:
        infile_txt = join(input_path, language, f"{part}.txt")
        outfile_txt = join(output_path, language, f"{part}.txt")
        infile_json = join(input_path, language, f"{part}_doc.json")
        outfile_json = join(output_path, language, f"{part}_doc.json")
        strip_question(infile_txt, outfile_txt)
        strip_question(infile_json, outfile_json)


if __name__ == "__main__":
    main()

