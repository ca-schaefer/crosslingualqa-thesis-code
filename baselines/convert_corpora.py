import argparse
import os

from convert_mlqa2xqa import read_mlqa
from convert_tydi2xqa import read_tydi
from convert_xquad2xqa import read_xquad, read_xquad_context
from data import save_xqa


LANGUAGE_MAP = {"th": "thai",
                "sw": "swahili",
                "te": "telugu",
                "fi": "finnish",
                "be": "bengali",
                "ru": "russian",
                "ja": "japanese",
                "ar": "arabic",
                "in": "indonesian",
                "ko": "korean",
                "en": "english",
                None: None
                }


def main():
    parser = argparse.ArgumentParser("Converting corpora (MLQA, XQUAD, TyDi QA) to XQA format")
    parser.add_argument("in_path", help="Path to corpus")
    parser.add_argument("out_path", help="Path where the result should be saved")
    parser.add_argument("-f", "--format", choices=["mlqa", "xquad", "tydi", "xquad-context"],
                        help="The format of the input corpus")
    parser.add_argument("-p", "--part", choices=["train", "dev", "test"])
    parser.add_argument("-l", "--language", default=None)

    args = parser.parse_args()
    part = args.part
    language = args.language

    if args.format == "mlqa":
        in_file = os.path.join(args.in_path, part, f"{part}-context-{language}-question-{language}.json")
        out_path = os.path.join(args.out_path, "MLQA")
        corpus = read_mlqa(in_file)
    elif args.format == "xquad":
        in_file = os.path.join(args.in_path, f"xquad.{language}.json")
        out_path = os.path.join(args.out_path, "XQUAD")
        corpus = read_xquad(in_file)
    elif args.format == "xquad-context":
        in_file = os.path.join(args.in_path, f"xquad.{language}.json")
        out_path = os.path.join(args.out_path, "XQUAD_context")
        corpus = read_xquad_context(in_file)
    elif args.format == "tydi":
        in_file = os.path.join(args.in_path, f"v1.0_tydiqa-v1.0-{part}.jsonl.gz")
        out_path = os.path.join(args.out_path, "TYDI")
        corpus = read_tydi(in_file, LANGUAGE_MAP[language])
    else:
        print("Wrong corpus format: ", args.format)
        raise NotImplementedError

    path = os.path.join(out_path, language)
    if not os.path.exists(path):
        os.makedirs(path)
    json_filename = os.path.join(path, f"{part}_doc.json")
    txt_filename = os.path.join(path, f"{part}.txt")
    save_xqa(corpus, json_filename, txt_filename)


if __name__ == "__main__":
    main()

