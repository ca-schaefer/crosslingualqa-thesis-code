import argparse
import sys


def compute_average_docqa_result(text):
    text = text.strip()
    print(text[:text.find("\n")])
    # skip to results
    pattern = "N Paragraphs EM     F1 "
    idx = text.find(pattern)
    if idx < 0:
        print("no valid evaluation given")
        return
    text = text[(idx + len(pattern)):]

    # read values
    lines = text.split("\n")
    em_list = []
    f1_list = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        n, current_em, current_f1 = line.split()
        em_list.append(float(current_em))
        f1_list.append(float(current_f1))

    print("Exact match: ", sum(em_list) / len(em_list))
    print("F1 Score: ", sum(f1_list) / len(f1_list))


PASTED = """
### en, dev, muse, tf-idf
N Paragraphs EM     F1    
1            0.2796 0.3269
2            0.2902 0.3410
3            0.2840 0.3347
4            0.2799 0.3328
5            0.2642 0.3147


### en, dev, muse, truncate
N Paragraphs EM     F1    
1            0.4040 0.4710
2            0.3691 0.4321
3            0.3247 0.3858
4            0.2878 0.3442
5            0.2580 0.3105
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, help="The evaluation file")
    args = parser.parse_args()

    if args.filename is not None:
        with open(args.filename) as fin:
            texts = [fin.read()]
    else:
        texts = PASTED.split("\n\n")

    for current_text in texts:
        compute_average_docqa_result(current_text)


