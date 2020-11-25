import html
import json
import os
from os.path import join
import re

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import wikipediaapi

# from get_wiki_articles import get_target_article
from get_BM25_documents import get_queries, save_closest, Article
from format_corpus import check_answer_in_doc, add_additional_docs, remove_first_paragraph, filter_without_document, \
    add_article_title_as_answer


def read_did_you_know(directory: str):
    # loop over files in directory
    # parse html
    questions = []  # (text, answer, answer title, answer link)
    (_, _, filenames) = next(os.walk(directory))
    for filename in filenames:
        print(filename)
        start = False
        with open(join(directory, filename)) as f:
            for line in f:
                if line.startswith('<div style="background:#FFFFFF; border:1px solid #AAAAAA; float:left; height:16em; '
                                   'margin: .5em 1em .5em 0; padding:.5em 1em; width: 300px;">'
                                   '<span style="font-weight:bold;">'):
                    start = True
                    continue
                if start:
                    match = re.match(r'<p>(.*)<a href="(.*)" title="(.*)">(.*)</a>(.*)$', line)
                    if match is not None:
                        groups = match.groups()
                        question_text = groups[0] + "<Query>" + groups[4]
                        answer = groups[3],
                        answer_title = groups[2]
                        answer_url = groups[1]
                        # print("question ", question_text)
                        # print("answer ", answer, answer_title, answer_url)
                        questions.append((question_text, answer, answer_title, answer_url))
    return questions


def save_questions_tsv(filename, questions):
    with open(filename, "w") as fout:
        for question in questions:
            fout.write("\t".join(question) + "\n")


def save_questions_json(filename, questions):
    with open(filename, "w") as fout:
        for question in questions:
            current = {"question": question[0],
                       "answers": question[1],
                       "source_title": "",
                       "source_url": "",
                       "target_title": question[2],
                       "target_url": question[3]}
            json.dump(current, fout, ensure_ascii=False)
            fout.write("\n")


def save_dev_txt(filename, questions):
    with open(filename, "w") as fout:
        for question in questions:
            json_object = {"answers": [question[1]], "question": question[0]}
            json.dump(json_object, fout, ensure_ascii=False)
            fout.write("\n")


class DummyPage(wikipediaapi.WikipediaPage):

    def __init__(self):
        pass

    def exists(self):
        return False


def get_target_article(infile, outfile):
    items = []
    with open(infile) as fin:
        for line in fin:
            items.append(json.loads(line))

    # wiki_en = wikipediaapi.Wikipedia('en')
    wiki_de = wikipediaapi.Wikipedia('de')
    with open(outfile, "w") as fout:
        for item in tqdm(items):
            # if i < 155:
            #     continue
            if item["target_title"]:
                page = wiki_de.page(item["target_title"])
            elif item["answers"]:
                page = wiki_de.page(item["answers"])
                item["target_title"] = item["answers"]
            else:
                page = DummyPage()
            if page.exists():
                item["target_text"] = page.text
                item["categories"] = [x.title() for x in page.categories]
            else:
                item["target_text"] = "page does not exist"
                item["categories"] = "page does not exist"
            json.dump(item, fout, ensure_ascii=False)
            fout.write("\n")
            fout.flush()


def get_10_closest_from_corpus(infile, queries):
    # load docs
    collected_articles = []
    with open(infile) as fin:
        for line in tqdm(fin):
            json_object = json.loads(line.strip())
            doc = json_object["target_text"]
            doc_id = json_object["target_title"]
            collected_articles.append(Article(doc_id, doc_id, doc, word_tokenize(doc, "german")))

    # compute bm25
    corpus = [art.word_list for art in collected_articles]
    bm25 = BM25Okapi(corpus)
    print("corpus indexed")

    closest = {query: [] for query in queries}
    for query in tqdm(queries):
        doc_scores = bm25.get_scores(query)
        tenth_best_score = sorted(doc_scores, reverse=True)[9]
        for idx, score in enumerate(doc_scores):
            if score >= tenth_best_score:
                closest[query].append((score, collected_articles[idx]))
        if len(closest[query]) < 10:
            print("Not enough closest queries")
            raise RuntimeError
    return closest


def format_html_umlauts(path, part):
    in_file = join(path, "ori.txt")
    out_file = join(path, "ori_cleaned.txt")
    with open(in_file) as fin:
        with open(out_file, "w") as fout:
            for line in fin:
                current = json.loads(line)
                current["question"] = unescape(current["question"])
                current["answers"] = [unescape(current["answers"][0][0])]
                json.dump(current, fout, ensure_ascii=False)
                fout.write("\n")

    in_file = join(path, "ori_single_doc_de.json")
    out_file = join(path, "ori_single_doc_de_cleaned.json")
    with open(in_file) as fin:
        with open(out_file, "w") as fout:
            for line in fin:
                current = json.loads(line)
                current["question"] = unescape(current["question"])
                current["answers"] = [unescape(current["answers"][0])]
                json.dump(current, fout, ensure_ascii=False)
                fout.write("\n")


def unescape(text: str):
    new_text = html.unescape(text)
    new_text = re.sub(r"<i>", "", new_text)
    new_text = re.sub(r"</i>", "", new_text)
    return new_text


def main():
    # data = read_did_you_know("did_you_know")
    path = "created_corpus_orig"
    dev_txt_file = join(path, "ori.txt")
    single_doc_file = f"created_corpus_orig/ori_single_doc_de.json"
    # save_questions_json(join(path, "saved_questions.json"), data)
    # get_target_article(join(path, "saved_questions.json"), single_doc_file)
    # save_dev_txt(dev_txt_file, data)
    # get other articles
    outfile = join(path, "corpus_ori_BM25_documents.json")
    # wiki_file = "/home/ca/wikipedia_de/dewiki-20200620-pages-meta-current1.xml-p1p262468.bz2"
    wiki_file = "/home/ca/wikipedia_de/dewiki-20200620-pages-articles-multistream.xml.bz2"

    # queries = get_queries(dev_txt_file)
    # data = get_10_closest_docs(wiki_file, queries)
    # data = get_10_closest_from_corpus(single_doc_file, queries)
    # save_closest(outfile, data)

    # format_html_umlauts(path, "ori")
    #
    # print("Filter without document")
    # filter_without_document(path, "ori", False)
    # print("Check answer in doc")
    # check_answer_in_doc(path, "ori")
    # print("Add additional docs")
    # add_additional_docs(path, "ori")
    # print("Remove first paragraph")
    # remove_first_paragraph(path, "ori")
    print("Add article title as answer")
    add_article_title_as_answer(path, "ori")
    pass


if __name__ == "__main__":
    main()
    # filename = "/home/ca/wikipedia_de/dewiki-20200620-pages-articles-multistream.xml.bz2"
    # find_did_you_know(filename)
    # q = read_did_you_know("did_you_know")
    # save_questions_json("saved_questions.json", q)
