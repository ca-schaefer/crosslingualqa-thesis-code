import argparse
from collections import Counter
import re
from os.path import join
import string
import sys

from bert_clone.tokenization import BasicTokenizer
from nltk import word_tokenize
from tqdm import tqdm

from data import load_xqa_wrapper
from documentqa_clone.docqa.data_processing.text_utils import NltkAndPunctTokenizer
from documentqa_clone.docqa.triviaqa.answer_detection import FastNormalizedAnswerDetector, compute_answer_spans_par

from evidence_corpus import XQAEvidenceCorpusTxt
from read_data import iter_question


BASIC_TOKENIZER = BasicTokenizer(do_lower_case=False)


def tokenize(sent, language, filter_punctuation):
    tokenized = None
    if language is None:
        tokenized = BASIC_TOKENIZER.tokenize(sent)
    else:
        tokenized = word_tokenize(sent, language)
    if filter_punctuation:
        tokenized = [token for token in tokenized if token[0] not in string.punctuation]
    return tokenized


def compute_num_question_tokens(data: dict, language: str, filter_punct: bool):
    total = 0
    for question in data:
        total += len(tokenize(question, language, filter_punct))
    return total / len(data)


def compute_num_article_tokes(data: dict, language: str, filter_punct: bool):
    total = 0
    num_articles = 0
    for item in tqdm(data.values()):
        articles = item.documents
        for art in articles:
            total += len(tokenize(art, language, filter_punct))
            num_articles += 1
    return total / len(data), total / num_articles


def compute_num_article_bytes(data: dict):
    total = 0
    num_articles = 0
    for item in data.values():
        articles = item.documents
        for art in articles:
            total += sys.getsizeof(art)
            num_articles += 1
    return total / len(data), total / num_articles


def compute_num_answer_tokens(data: dict, language: str, filter_punct: bool):
    # averaged
    total = 0
    num_answers = 0
    for item in data.values():
        for answer in item.gold:
            total += len(tokenize(answer, language, filter_punct))
            num_answers += 1
    return total / num_answers


def compute_num_answer_bytes(data: dict):
    total = 0
    num_answers = 0
    for item in data.values():
        for answer in item.gold:
            total += sys.getsizeof(answer)
            num_answers += 1
    return total / num_answers


def compute_num_passage_candidates(data: dict):
    total_candidates = 0
    num_articles = 0
    for item in data.values():
        for article in item.documents:
            for answer in item.gold:
                total_candidates += article.count(answer)
            num_articles += 1
    return total_candidates / len(data), total_candidates / num_articles


def get_stopwords(path, language, filter_punct):
    xqa_data = load_xqa_wrapper(path, "dev")
    all_text = []
    for item in xqa_data.values():
        for doc in item.documents:
            all_text.extend(tokenize(doc, LANGUAGES[language], filter_punct))
    counter = Counter(all_text)
    return {x[0] for x in counter.most_common(100)}


def compute_lexical_overlap(path: str, language: str, n_processes: int, filter_punct: bool):
    """computes the average number of tokens in common between the question and a 200 - character window around
    the answer span"""
    total = 0
    docs_with_answers = 0
    docs_without_answers = 0

    train_files = dict(
                      dev=join(path, "qa", "dev.json"),
                      # train=join(path, "qa", "train.json"),
                      # test=join(path, "qa", "test.json")
                  )
    tokenizer = NltkAndPunctTokenizer()
    answer_detector = FastNormalizedAnswerDetector()

    file_map = {}

    stop_words = get_stopwords(path, language, filter_punct)
    print(stop_words)

    for name, filename in train_files.items():
        print("Loading %s questions" % name)
        questions = list(iter_question(filename, file_map))

        for q in questions:
            q.docs = [x for x in q.docs if x.doc_id in file_map]

        corpus = XQAEvidenceCorpusTxt("XQA_original", language, file_map)
        questions = compute_answer_spans_par(questions, corpus, tokenizer, answer_detector, n_processes)

        base_path = join(path, "tokenized/evidence")
        for q in tqdm(questions):  # Sanity check, we should have answers for everything (even if of size 0)
            question_words = set(q.question)
            question_words = {word for word in question_words if word not in string.punctuation}
            if q.answer is None:
                # print("No answer")
                continue

            overlaps = []
            for doc in q.docs:
                if doc.doc_id in file_map:
                    if doc.answer_spans is None:
                        print("No answer spans: ", doc.doc_id)
                    else:
                        spans = doc.answer_spans
                        # Load doc
                        path_id = "_".join(doc.doc_id.split("_")[:2])
                        filename = doc.doc_id + ".txt"
                        with open(join(base_path, path_id, filename)) as fin:
                            flat_text = fin.read()

                        if spans.shape >= (1, 2):
                            current_overlaps = []
                            for span in spans:
                                start_idx = span[0]
                                end_idx = span[1] + 1
                                # span start - 100 to span end + 100
                                window = (max(start_idx - 100, 0), min(end_idx + 100, len(flat_text)))
                                content_words = flat_text[window[0]:window[1]].split()
                                # remove punctuation
                                content_words = {word for word in content_words if word not in string.punctuation}
                                content_words.difference_update(stop_words)
                                overlap = len(content_words.intersection(question_words))
                                current_overlaps.append(overlap)
                            overlaps.append(max(current_overlaps))

                        else:
                            # print("No answer span ", spans)
                            pass
            if overlaps:
                total += max(overlaps)
                docs_with_answers += 1
            else:
                # print(q.question, " has no answers")
                docs_without_answers += 1
        print("Total overlap ", total)
        print("with answers ", docs_with_answers)
        print("without answers ", docs_without_answers)
        print("Average overlap ", total / docs_with_answers)
    return total / docs_with_answers


def compute_answer_spans(data: dict):
    spans = {}
    for item in data.values():
        question = item.question
        articles = item.documents
        flat_articles = "\n".join(articles)
        current_spans = []
        for answer in item.answers:
            starts = [match.start() for match in re.finditer(answer, flat_articles)]
            ends = [start + len(answer) for start in starts]
            current_spans.extend(zip(starts, ends))
        spans[question] = current_spans
    return spans


LANGUAGES = {"de": "german",
             "en": "english",
             "ta": None,
             "fr": "french",
             "pl": "polish",
             "pt": "portuguese",
             "ru": "russian",
             "uk": None,
             "zh": None
             }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", choices=["de", "en", "all"], required=True)
    parser.add_argument("-p", "--part", choices=["train", "dev", "test", "all"])
    parser.add_argument("-f", "--filter", action="store_true")
    args = parser.parse_args()

    language_list = [args.language]
    if args.language == "all":
        language_list = list(LANGUAGES.keys())

    if args.part == "all":
        part_list = ["train", "dev", "test"]
    else:
        part_list = [args.part]
    for language in language_list:
        print("language: ", language)
        for part in part_list:
            print("part: ", part)
            path = f"/home/ca/Documents/Uni/Masterarbeit/data/XQA_original/{language}/"
            nltk_language = LANGUAGES[language]
            data = load_xqa_wrapper(path, part)
            print("data loaded.")

            print("Num question tokens: ", compute_num_question_tokens(data, nltk_language, args.filter))
            print("Num article tokens: ", compute_num_article_tokes(data, nltk_language, args.filter))
            print("Num article bytes: ", compute_num_article_bytes(data))
            print("Lexical overlap: ", compute_lexical_overlap(path, language, 2, False))
            print("Num answer tokens: ", compute_num_answer_tokens(data, nltk_language, args.filter))
            print("Num answer bytes: ", compute_num_answer_bytes(data))
            print("Num passage candidates", compute_num_passage_candidates(data))
            print()

