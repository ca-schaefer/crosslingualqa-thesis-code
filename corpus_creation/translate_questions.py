import json

import googletrans
from tqdm import tqdm


def translate(infile, outfile):
    translator = googletrans.Translator()
    with open(infile) as fin:
        with open(outfile, "w") as fout:
            for line in tqdm(fin):
                line = line.strip()
                if not line:
                    continue
                try:
                    current = json.loads(line)
                except json.decoder.JSONDecodeError as e:
                    print("line: ", line)
                    raise e
                question = current["question"]
                current["translated_question"] = translator.translate(question, dest="de").text
                answers = current["answers"]
                current["translated_answers"] = [translator.translate(answer, dest="de").text for answer in answers]
                json.dump(current, fout, ensure_ascii=False)
                fout.write("\n")
                fout.flush()


if __name__ == "__main__":
    infile = "/home/ca/Documents/Uni/Masterarbeit/crosslingual-qa-scripts/corpus_creation/tail_train_urls.json"
    outfile = "/home/ca/Documents/Uni/Masterarbeit/crosslingual-qa-scripts/corpus_creation/train_translated_4.json"
    translate(infile, outfile)
