import argparse
import logging
import random
import os
from typing import List

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import spacy
from tqdm import tqdm

from documentqa_clone.docqa.triviaqa.trivia_qa_eval import exact_match_score, f1_score, metric_max_over_ground_truths

from data import load_xqa_wrapper


def ne_with_wordoverlap(documents: List[str], nlp, question: str):
    """Take sentence with highest word overlap with the question, returns NE from that that is not in question"""
    sentences = []
    question_set = set(word_tokenize(question))
    for document in documents:
        sentences.extend(sent_tokenize(document))

    # get sentence with best overlap
    best_overlap = (0, "")
    for sentence in sentences:
        current_overlap = len(set(word_tokenize(sentence)).intersection(question_set))
        if current_overlap > best_overlap[0]:
            best_overlap = (current_overlap, sentence)

    # get NE from that sentence
    doc = nlp(best_overlap[1])
    entities_not_in_question = [ent.text for ent in doc.ents if not ent.text in question]
    if len(entities_not_in_question) > 0:
        return random.choice(entities_not_in_question)
    # print("Not found ", doc.ents, question)
    return ""


def noun_with_wordoverlap(documents: List[str], nlp, question: str):
    """Take sentence with highest word overlap with the question, returns NE from that that is not in question"""
    sentences = []
    question_set = set(word_tokenize(question))
    for document in documents:
        sentences.extend(sent_tokenize(document))

    # get sentence with best overlap
    best_overlap = (0, "")
    for sentence in sentences:
        current_overlap = len(set(word_tokenize(sentence)).intersection(question_set))
        if current_overlap > best_overlap[0]:
            best_overlap = (current_overlap, sentence)

    # get NE from that sentence
    doc = nlp(best_overlap[1])
    nouns = [token.text for token in doc if (token.pos_ == "NOUN" or token.pos_ == "PROPN")]
    nouns_not_in_question = [noun for noun in nouns if (noun not in question)]

    if len(nouns_not_in_question) > 0:
        return random.choice(nouns_not_in_question)
    print(nouns_not_in_question, question, best_overlap[1])
    return ""


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Evaluate a baseline model')
    parser.add_argument("corpus", help="Path to the eval data")
    parser.add_argument("-m", "--model", choices=["ne", "noun"])
    parser.add_argument("-n", "--n_best", type=int, help="Number of documents where the answer should be checked")
    parser.add_argument("--paragraph_output", action="store_true",
                        help="Save fine grained results for each paragraph in csv format")
    parser.add_argument("-l", "--language", choices=["de", "en"], default="de",
                        help="Language of the used preprocessing system")

    parser.add_argument("-p", "--part", choices=["train", "dev", "test", "all"])

    args = parser.parse_args()
    logging.info(str(args))

    # load data
    data = load_xqa_wrapper(os.path.join(args.corpus, args.language), args.part)

    # build model
    if args.model == "ne":
        predict = ne_with_wordoverlap
    elif args.model == "noun":
        predict = noun_with_wordoverlap
    else:
        raise NotImplementedError("Baseline models are 'ne', 'noun'")


    if args.language == "de":
        nlp = spacy.load("de_core_news_sm")
    elif args.language == "en":
        nlp = spacy.load("en_core_web_sm")
    else:
        raise NotImplementedError("Only German (de) and English (en) implemented")

    # run baseline
    is_writing = False
    if args.paragraph_output:
        is_writing = True
        corpus_name = args.corpus.split("/")[-1]
        output_file = open(f"paragraph_output_{corpus_name}_{args.language}_{args.part}_overlap_n{args.n_best}.txt", "w")
    correct = 0
    incorrect = 0
    f1_collection = []
    em_collection = []
    for question, item in tqdm(list(data.items())):
        if (args.n_best is not None) and (args.n_best < len(item.documents)):
            context = item.documents[:args.n_best]
        else:
            context = item.documents
        prediction = predict(context, nlp, item.question)
        logging.info(f"Question: {item.question} \t pred: {prediction}\t gold: {item.gold}")
        if is_writing:
            output_file.write(f"Question: {item.question} \t pred: {prediction}\t gold: {item.gold}\n")

        f1_collection.append(metric_max_over_ground_truths(f1_score, prediction, item.gold))
        em_collection.append(metric_max_over_ground_truths(exact_match_score, prediction, item.gold))

        if prediction in item.gold:
            correct += 1
        else:
            incorrect += 1
    print(f"Accuracy: {correct / len(data)}")

    print("F1:", np.mean(f1_collection))
    print("EM:", np.mean(em_collection))
    if is_writing:
        output_file.write(f"Accuracy: {correct / len(data)}\n")
        output_file.write(f"F1: {np.mean(f1_collection)}\n")
        output_file.write(f"EM: {np.mean(em_collection)}\n")
        output_file.close()

