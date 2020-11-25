from bz2 import BZ2File
import json
import xml.etree.ElementTree as etree

from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from wiki_dump_reader import Cleaner, iterate


class Article(object):

    def __init__(self, title, id, text, word_list):
        self.title = title
        self.id = id
        self.text = text
        self.word_list = word_list

    def __repr__(self):
        res = "Article: "
        res += self.title
        res += " - " + str(self.id)
        res += f" with {len(self.word_list)} words"
        return res


def get_queries(filename):
    queries = []
    with open(filename) as fin:
        for line in fin:
            json_object = json.loads(line.strip())
            queries.append(tuple(word_tokenize(json_object["question"], "german")))
    return queries


def clean_article(text: str):
    text = text.replace("[", "")
    text = text.replace("]", "")
    return text


def get_10_closest_docs(infile, queries):
    closest = {query: {} for query in queries}
    # cleaner = Cleaner()
    # for title, text in iterate(infile):
    #     text = cleaner.clean_text(text)
    #     cleaned_text, _ = cleaner.build_links(text)
    #     print(cleaned_text)
    #     return

    def strip_tag_name(t):
        """From https://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html"""
        t = elem.tag
        idx = t.rfind("}")
        if idx != -1:
            t = t[idx + 1:]
        return t

    page_count = 0
    collected_articles = []
    with BZ2File(infile, "rb") as bzfin:
        for event, elem in tqdm(etree.iterparse(bzfin, events=('start', 'end'))):
            tag_name = strip_tag_name(elem.tag)
            # print(tag_name)

            if event == 'start':
                if tag_name == 'page':
                    title = ''
                    id = -1
                    in_revision = False
                    ns = 0
                elif tag_name == 'revision':
                    # Do not pick up on revision id's
                    in_revision = True
            else:
                if tag_name == 'title':
                    title = elem.text
                elif tag_name == 'id' and not in_revision:
                    id = int(elem.text)
                elif tag_name == "text":
                    article = elem.text
                    if not isinstance(article, str):
                        print("article ", article)
                        print("title ", title)
                        print(elem)
                        continue
                    collected_articles.append(Article(title, id, article, word_tokenize(article, "german")))
                elif tag_name == 'page':
                    page_count += 1
                    if page_count > 0 and page_count % 1000 == 0:  # change to 1000?
                        bm25 = BM25Okapi([art.word_list for art in collected_articles])
                        for query in closest:
                            if closest[query]:
                                minimal_score = min(closest[query])
                            else:
                                minimal_score = -1
                            # print(minimal_score)
                            doc_scores = bm25.get_scores(query)
                            # print(doc_scores)
                            for idx, score in enumerate(doc_scores):
                                if score >= minimal_score:
                                    closest[query][score] = collected_articles[idx]
                            # prune score dict again
                            if (len(closest[query])) > 10:
                                tenth_best_score = sorted(closest[query].keys(), reverse=True)[9]
                                for score in list(closest[query].keys()):
                                    if score < tenth_best_score:
                                        closest[query].pop(score)
                        collected_articles = []
                        save_closest("tmp_train.txt", closest)
                        print("page_count ", page_count)
                elem.clear()
    return closest


def save_closest(filename: str, closest: dict):
    with open(filename, "w") as fout:
        for query in closest:
            json_list = []
            for score, article in closest[query]:
                json_list.append({"score": score,
                               "title": article.title,
                               "text": article.text,
                               "id": article.id})
            json_object = {"question": " ".join(query),
                           "documents": json_list}
            json.dump(json_object, fout, ensure_ascii=False)
            fout.write("\n")


def get_10_closest_from_corpus(infile, queries):
    # load docs
    collected_articles = []
    with open(infile) as fin:
        for line in tqdm(fin):
            json_object = json.loads(line.strip())
            # [{"id": [0, 0],
            #   "question": "Der halluzinogene Pilz <Query> \"\" wurde erstmals in einem tropischen Regenwald in der
            #   Region Uxpanapa in Veracruz im Südosten Mexikos entdeckt.",
            #   "document": "page does not exist", "document_id": "Psilocybe naematoliformis"}]
            doc = json_object[0]["document"]
            doc_id = json_object[0]["document_id"]
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


if __name__ == "__main__":
    #part = "dev"
    part = "train"
    outfile = f"updated_{part}_BM25_documents.json"
    # wiki_file = "/home/ca/wikipedia_de/dewiki-20200620-pages-meta-current1.xml-p1p262468.bz2"
    wiki_file = "/home/ca/wikipedia_de/dewiki-20200620-pages-articles-multistream.xml.bz2"
    query_file = f"created_corpus/filtered_{part}_de.txt"
    # doc_file = f"{part}_doc_de.json"
    doc_file = f"created_corpus/filtered_{part}_single_doc_de.json"
    queries = get_queries(query_file)
    # data = get_10_closest_docs(wiki_file, queries)
    data = get_10_closest_from_corpus(doc_file, queries)
    save_closest(outfile, data)
