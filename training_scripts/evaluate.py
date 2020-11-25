"""
Common evaluation scripts for all models.

Creates copies of config and takes care of saving all information.
"""

import argparse
import configparser
import os
import subprocess
import sys
import time

from config import SAVED_EVALUATION_PATH, SIMPLE_BASELINE_PATH, XQA_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation script")
    parser.add_argument("config", help="The current configuration file")

    args = parser.parse_args()

    # Get correct configuration
    config = configparser.ConfigParser()
    config.read(args.config)
    data_path = config["COMMON"]["DataPath"]
    origin = config["COMMON"]["CorpusOrigin"]
    language = config["COMMON"]["CorpusLanguage"]
    part = config["COMMON"]["CorpusPart"]
    model_dir = config["COMMON"]["TrainedModelDir"]
    model_type = config["COMMON"]["Type"]

    # make new directory in evaluations with date
    new_dir_name = "_".join([model_type, origin, language, part, time.strftime('%Y%m%d-%H%M%S')])
    new_directory = os.path.join(SAVED_EVALUATION_PATH, new_dir_name)
    os.makedirs(new_directory)

    # save configuration in that directory
    with open(os.path.join(new_directory, "config.ini"), "w", encoding="utf-8") as fout:
        config.write(fout)

    if model_type in {"documentqa", "bert"}:
        old_model_config = os.path.join(model_dir, "config.ini")
        new_model_config = os.path.join(new_directory, "model_config.ini")
        os.popen("cp " + old_model_config + " " + new_model_config)

    out_file = os.path.join(new_directory, "results.txt")
    if model_type == "simple":
        nbest = config["SIMPLE"]["NBest"]
        model = config["SIMPLE"]["Model"]
        filename = os.path.join(data_path, language)
        detailed_out_file = os.path.join(new_directory, "per_question_result.txt")
        arguments = ["python", SIMPLE_BASELINE_PATH, filename, "-m", model, "-n", nbest, "-l", language,
                     "-p", detailed_out_file]
        if part == "test":
            arguments.append("--test")
        output = subprocess.check_output(arguments)
        with open(out_file, "wb") as fout:
            fout.write(output)
    elif model_type == "documentqa":
        # Load model
        paragraph_output = os.path.join(new_directory, config["DOCUMENTQA"]["ParagraphOutput"])
        official_output = os.path.join(new_directory, config["DOCUMENTQA"]["OfficialOutput"])
        no_ema = True if config["DOCUMENTQA"]["NoEma"] == "True" else False
        n_processes = config["DOCUMENTQA"]["NProcesses"]
        step = config["DOCUMENTQA"]["Step"]
        n_sample = config["DOCUMENTQA"]["NSample"]
        async_ = config["DOCUMENTQA"]["ASYNC"]
        tokens = config["DOCUMENTQA"]["Tokens"]
        n_paragraphs = config["DOCUMENTQA"]["NParagraphs"]
        filter_ = config["DOCUMENTQA"]["Filter"]
        batch_size = config["DOCUMENTQA"]["BatchSize"]
        max_answer_len = config["DOCUMENTQA"]["MaxAnswerLen"]
        dump_data_pickle_only = True if config["DOCUMENTQA"]["DumpDataPickleOnly"] == "True" else False
        embeddings = config["DOCUMENTQA"]["Embeddings"]
        embedding_size = config["DOCUMENTQA"]["EmbeddingSize"]

        run_arguments = ["python", os.path.join(XQA_PATH, "document_qa_eval.py"), model_dir, origin, language, part, 
                         "--paragraph_output", paragraph_output,
                         "--official_output", official_output, "--n_processes", n_processes, "--async_", async_,
                         "--tokens", tokens, "--n_paragraphs", n_paragraphs, "--filter", filter_,
                         "--batch_size", batch_size, "--max_answer_len", max_answer_len, "--embeddings", embeddings,
                         "--embedding_size", embedding_size]
        if no_ema:
            run_arguments.append("--no_ema")
        if step != "None":
            run_arguments.extend(["--step", step])
        if n_sample != "None":
            run_arguments.extend(["--n_sample", n_sample])
        if dump_data_pickle_only:
            run_arguments.append("--dump_data_pickle_only")
        
        try: 
            output = subprocess.check_output(run_arguments)
        except subprocess.CalledProcessError as e:
            print(e.output)
            sys.exit(0)
        with open(out_file, "wb") as fout:
            fout.write(output)

    elif model_type == "bert":
        n_paragraphs = config["BERT"]["NParagraphs"]
        tokens = config["BERT"]["Tokens"]
        vocab_file = config["BERT"]["VocabFile"]
        bert_config_file = config["BERT"]["BertConfigFile"]
        do_lower_case = True if config["BERT"]["DoLowerCase"] == "True" else False
        max_seq_length = config["BERT"]["MaxSeqLength"]
        max_query_length = config["BERT"]["MaxQueryLength"]
        predict_batch_size = config["BERT"]["PredictBatchSize"]
        init_checkpoint = config["BERT"]["InitCheckpoint"]
        tagging = config["BERT"]["Tagging"]

        # Determine name of cached input file (named "output")
        preprocessed_eval_file = "../data/%s_%s_%s_output_%s_%s.json" % (origin, language, tagging, part, n_paragraphs)

        # Check if prediction file is already cached, if not create
        if not os.path.isfile(preprocessed_eval_file):
            pkl_file = "../data/%s_%s_%s_%s_%s.pkl" % (origin, language, tagging, part, n_paragraphs)
            cache_arguments = ["python", os.path.join(XQA_PATH, "cache_test.py"), origin, language, part, pkl_file,
                               "--n_paragraphs", n_paragraphs, "--tokens", tokens]
            print("Cache data: ", cache_arguments)
            subprocess.run(cache_arguments)
            dump_arguments = ["python", os.path.join(XQA_PATH, "dump_preprocessed_eval.py"), "--input_file", pkl_file,
                              "--output_file", preprocessed_eval_file, "--language", language, "--tagging", tagging]
            print("Dump data: ", dump_arguments)
            subprocess.run(dump_arguments)

        # Run actual evaluation
        question_prediction_file = os.path.join(new_directory, "%s-%s-%s-%s-question-output.txt" %
                                                (origin, language, part, n_paragraphs))
        paragraph_prediction_file = os.path.join(new_directory, "%s-%s-%s-%s-paragraph-output.txt" %
                                                (origin, language, part, n_paragraphs))
        run_arguments = ["python", os.path.join(XQA_PATH, "run_bert_open_qa_eval.py"), "--vocab_file", vocab_file,
                         "--bert_config_file", bert_config_file, "--init_checkpoint", init_checkpoint,
                         "--predict_file", preprocessed_eval_file, "--predict_batch_size", predict_batch_size,
                         "--max_seq_length", max_seq_length, "--max_query_length", max_query_length,
                         "--model_dir", model_dir, "--do_lower_case", str(do_lower_case),
                         "--question_prediction_file", question_prediction_file,
                         "--paragraph_prediction_file", paragraph_prediction_file]
        print("Start evaluation: ", run_arguments)
        subprocess.run(run_arguments)
        name = "-".join(preprocessed_eval_file.split("_")[2:-1])
        metric_arguments = ["python", os.path.join(XQA_PATH, "get_evaluation_metric_for_bert_result.py"),
                            "--input_file", preprocessed_eval_file, "--prediction_file", question_prediction_file]
        output = subprocess.check_output(metric_arguments)
        with open(out_file, "wb") as fout:
            fout.write(output)

    else:
        raise NotImplementedError("Model type: %s" % model_type)
