"""
Simple baselines for XQA
"""

import argparse
from collections import Counter
import logging
import random
import os
from typing import List

import numpy as np
import spacy
from tqdm import tqdm

# sys.path = ["/home/ca/Documents/Uni/Masterarbeit/XQA/documentqa/"] + sys.path
# sys.path.append("/home/ca/Documents/Uni/Masterarbeit/XQA/documentqa/")

from documentqa_clone.docqa.triviaqa.trivia_qa_eval import exact_match_score, f1_score, metric_max_over_ground_truths

from data import load_xqa_wrapper


def predict_noun(documents: List[str], nlp):
    """Returns the most frequent noun in a given document list"""
    doc = nlp("".join(documents))
    noun_list = [token.text for token in doc if (token.pos_ == "NOUN" or token.pos_ == "PROPN")]
    counts = Counter(noun_list)
    if counts:
        return counts.most_common(1)[0][0]
    return ""


def predict_noun_phrase(documents: List[str], nlp):
    """Returns the most frequent noun phrase in a given document list"""
    pass
    # merge_noun_chunks function
    #
    # Merge noun chunks into a single token. Also available via the string name "merge_noun_chunks".
    # After initialization, the component is typically added to the processing pipeline using nlp.add_pipe
    #
    # .
    # Example
    #
    # texts = [t.text for t in nlp("I have a blue car")]
    # assert texts == ["I", "have", "a", "blue", "car"]
    #
    # merge_nps = nlp.create_pipe("merge_noun_chunks")
    # nlp.add_pipe(merge_nps)
    #
    # texts = [t.text for t in nlp("I have a blue car")]
    # assert texts == ["I", "have", "a blue car"]
    #
    # Since noun chunks require part-of-speech tags and the dependency parse, make sure to add this component after
    # the "tagger" and "parser" components.
    # By default, nlp.add_pipe will add components to the end of the pipeline and after all other components.


def predict_ne(documents: List[str], nlp):
    """Returns the most frequent Named Entity in a given document list"""
    doc = nlp("".join(documents))
    ne_list = [ent.text for ent in doc.ents]
    counts = {ent: ne_list.count(ent) for ent in ne_list}
    logging.debug(f"Counts {sorted(counts.items(), key=lambda x: x[1])}")
    try:
        most_frequent = max(counts.items(), key=lambda x: x[1])
        return most_frequent[0]
    except ValueError as e:
        return ""


def random_ne(documents: List[str], nlp):
    """Returns a random Named Entity from a given document list"""
    doc = nlp("".join(documents))
    ne_list = [ent.text for ent in doc.ents]
    # TODO: convert to set so every NE occurs only once or leave more frequent NEs more frequent?
    if ne_list:
        return random.choice(ne_list)
    return ""


def first_ne(documents: List[str], nlp):
    """Returns the first Named Entity from a list of documents"""
    doc = nlp(documents[0])
    if len(doc.ents) > 0:
        return doc.ents[0].text
    return ""


def predict_ngram(documents: List[str], nlp):
    pass


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Evaluate a baseline model')
    parser.add_argument("corpus", help="Path to the eval data")
    parser.add_argument("-m", "--model", choices=["ne", "noun", "n-gram", "random-ne", "first-ne"])
    parser.add_argument("-n", "--n_best", type=int, help="Number of documents where the answer should be checked")
    parser.add_argument("--paragraph_output", action="store_true",
                        help="Save fine grained results for each paragraph in csv format")
    # parser.add_argument("-o", "--official_output", type=str, help="Build an official output file with the model's"
    #                                                               " most confident span for each (question, doc) pair")
    parser.add_argument("-l", "--language", choices=["de", "en"], default="de",
                        help="Language of the used preprocessing system")

    parser.add_argument("-p", "--part", choices=["train", "dev", "test", "all"])

    args = parser.parse_args()
    logging.info(str(args))

    # load data
    data = load_xqa_wrapper(os.path.join(args.corpus, args.language), args.part)

    # build model
    if args.model == "ne":
        predict = predict_ne
    elif args.model == "noun":
        predict = predict_noun
    elif args.model == "n-gram":
        predict = predict_ngram
    elif args.model == "random-ne":
        predict = random_ne
    elif args.model == "first-ne":
        predict = first_ne
    else:
        raise NotImplementedError("Baseline models are 'ne', 'noun', 'n-gram'")

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
        output_file = open(f"paragraph_output_{corpus_name}_{args.language}_{args.part}_{args.model}_n{args.n_best}.txt", "w")
    correct = 0
    incorrect = 0
    f1_collection = []
    em_collection = []
    for question, item in tqdm(list(data.items())):
        if (args.n_best is not None) and (args.n_best < len(item.documents)):
            context = item.documents[:args.n_best]
        else:
            context = item.documents
        prediction = predict(context, nlp)
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
