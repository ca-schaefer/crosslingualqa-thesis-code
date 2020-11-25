from ast import literal_eval
import json

from tqdm import tqdm
import wikipediaapi


class Article(object):

    def __init__(self, question: str, answers:tuple, source_title: str, source_url: str, target_title: str,
                 target_url: str, same_target: str):
        self.question = question
        self.answers = answers
        self.source_title = source_title
        self.source_url = source_url
        self.target_title = target_title
        self.target_url = target_url

    def set_target_text(self, text: str):
        self.target_text = text


def get_corpus_urls(filename):
    pass


def get_answers(filename):
    answers = {}
    with open(filename) as fin:
        for line in fin:
            json_object = json.loads(line.strip())
            raw_answers = json_object["answers"]
            raw_answers.sort(key=lambda x: len(x))
            answers[tuple(raw_answers)] = json_object["question"]
    return answers


def get_original_articles(answers, part, target_language="de"):
    wiki_en = wikipediaapi.Wikipedia('en')
    url_map = {}
    translation_map = {}
    not_found_pages = []

    with open(f"{part}_url_map.tsv", "w") as url_file:
        for answer_set in tqdm(answers):
            found = False
            target_page = None
            source_page = None
            old_source_page = None
            for answer_option in answer_set:
                source_page = wiki_en.page(answer_option)
                if source_page.exists():
                    found = True
                    old_source_page = source_page
                    url_map[answer_set] = source_page.fullurl
                    langlinks = source_page.langlinks
                    same_target = False
                    if target_language in langlinks:
                        same_target = True
                        target_page = langlinks[target_language]
                        translation_map[answer_set] = target_page
                else:
                    source_page = old_source_page
            if source_page is None:
                source_title = None
                source_url = None
            else:
                source_title = source_page.title
                source_url = source_page.fullurl
            if target_page is None:
                target_title = None
                target_url = None
            else:
                target_title = target_page.title
                target_url = target_page.fullurl
            line = "\t".join(str(x) for x in (answers[answer_set], answer_set, source_title, source_url, target_title,
                                              target_url, same_target))
            url_file.write(line + "\n")
            url_file.flush()
            if not found:
                not_found_pages.append(answer_set)
    print(f"{len(answers)} pages processed")
    print(f"{len(url_map)} pages found in English Wikipedia: {len(url_map) / len(answers)}")
    print(f"{len(translation_map)} pages found in German Wikipedia: {len(translation_map) / len(answers)}")
    print(f"Not found pages {not_found_pages}")


def read_url_map(filename):
    urls = []
    with open(filename) as fin:
        for line in fin:
            parts = line.strip().split("\t")
            parts[1] = literal_eval(parts[1])
            urls.append(Article(*parts))
    return urls


def get_target_article(infile, outfile):
    urls = read_url_map(infile)
    # wiki_en = wikipediaapi.Wikipedia('en')
    wiki_de = wikipediaapi.Wikipedia('de')
    with open(outfile, "w") as fout:
        for url in tqdm(urls):
            current_json = {"question": url.question,
                            "answers": url.answers,
                            "source_title": url.source_title,
                            "source_url": url.source_url,
                            "target_title": url.target_title,
                            "target:url": url.target_url}
            if url.target_title == "None":
                current_json["target_text"] = "page does not exist"
                current_json["categories"] = "page does not exist"
            else:
                page = wiki_de.page(url.target_title)
                if page.exists():
                    current_json["target_text"] = page.text
                    current_json["categories"] = [x.title() for x in page.categories]
                else:
                    current_json["target_text"] = "page does not exist"
                    current_json["categories"] = "page does not exist"
                    # current_json["sections"] = page.sections
            json.dump(current_json, fout, ensure_ascii=False)
            fout.write("\n")
            fout.flush()


def main():
    # part = "train_tail"
    # corpus = f"/home/ca/Documents/Uni/Masterarbeit/data/XQA_original/en/{part}.txt"
    #
    # answers = get_answers(corpus)
    # print(len(answers))
    # get_original_articles(answers, part)
    # infile = "/home/ca/Documents/Uni/Masterarbeit/crosslingual-qa-scripts/corpus_creation/2020_06_09/full_train_url_map_2.tsv"
    # outfile = "/home/ca/Documents/Uni/Masterarbeit/crosslingual-qa-scripts/corpus_creation/train_urls.json"
    # get_target_article(infile, outfile)
    pass


if __name__ == "__main__":
    main()

