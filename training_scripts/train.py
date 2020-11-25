import argparse
import configparser
import os
from os.path import join
import subprocess
import time

from config import SAVED_MODELS_PATH, XQA_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script")
    parser.add_argument("config", help="The current configuration file")

    args = parser.parse_args()

    start_time = time.time()

    # Get correct configuration
    config = configparser.ConfigParser()
    config.read(args.config)
    origin = config["COMMON"]["CorpusOrigin"]
    language = config["COMMON"]["CorpusLanguage"]
    model_type = config["COMMON"]["Type"]

    # make new directory in saved models with date
    new_dir_name = "_".join([model_type, origin, language, time.strftime('%Y%m%d-%H%M%S')])
    new_directory = join(SAVED_MODELS_PATH, new_dir_name)
    os.makedirs(new_directory)
    print("Prepared training model " + new_dir_name)
    print("in Working directory ", os.getcwd())

    # save configuration in that directory
    with open(join(new_directory, "config.ini"), "w", encoding="utf-8") as fout:
        config.write(fout)

    if model_type == "documentqa":
        mode = config["DOCUMENTQA"]["Mode"]
        paragraph_size = config["DOCUMENTQA"]["ParagraphSize"]
        n_processes = config["DOCUMENTQA"]["NProcesses"]
        embeddings = config["DOCUMENTQA"]["Embeddings"]
        epochs = config["DOCUMENTQA"]["Epochs"]
        train_batch_size = config["DOCUMENTQA"]["TrainBatchSize"]
        eval_batch_size = config["DOCUMENTQA"]["EvalBatchSize"]
        embedding_size = config["DOCUMENTQA"]["EmbeddingSize"]
        eval_period = config["DOCUMENTQA"]["EvalPeriod"]
        save_period = config["DOCUMENTQA"]["SavePeriod"]
        num_eval_samples = config["DOCUMENTQA"]["NumEvalSamples"]
        model_dimension = config["DOCUMENTQA"]["ModelDimension"]
        char_th = config["DOCUMENTQA"]["CharTH"]
        max_checkpoints_to_keep = config["DOCUMENTQA"]["MaxCheckPointsToKeep"]
        checkpoint = config["DOCUMENTQA"]["Checkpoint"]

        ablation_arguments = ["python", join(XQA_PATH, "ablate_xqa.py"),  origin, language, mode, new_directory,
                              "-t", paragraph_size, "-n", n_processes, "-e", embeddings, "--epochs", epochs,
                              "--train_batch_size", train_batch_size, "--eval_batch_size", eval_batch_size,
                              "--embedding_size", embedding_size, "--eval_period", eval_period,
                              "--save_period", save_period, "--num_eval_samples", num_eval_samples,
                              "--model_dimension", model_dimension, "--char_th", char_th,
                              "--max_checkpoints_to_keep", max_checkpoints_to_keep]
        if checkpoint != "None":
            ablation_arguments.extend(["--checkpoint", checkpoint])
        print("Start training with arguments " + str(ablation_arguments))
        subprocess.run(ablation_arguments)
    elif model_type == "bert":
        mode = config["BERT"]["Mode"]
        preprocess_epochs = config["BERT"]["PreprocessEpochs"]
        tagging = config["BERT"]["Tagging"]
        preprocessed_train_file = "../data/%s_%s_%s_train_output.json" % (origin, language, tagging)
        if not os.path.isfile(preprocessed_train_file):
            pkl_file = "../data/train_%s_%s_%s.pkl" % (origin, language, tagging)
            cache_arguments = ["python", join(XQA_PATH, "cache_train.py"), origin, language, mode, pkl_file]
            print("Cache training data: ", cache_arguments)
            subprocess.run(cache_arguments)
            dump_arguments = ["python", join(XQA_PATH, "dump_preprocessed_train.py"), "--input_file", pkl_file,
                              "--output_train_file", preprocessed_train_file, "--num_epoch", preprocess_epochs,
                              "--language", language, "--tagging", tagging]
            print("Dump training data: ", dump_arguments)
            subprocess.run(dump_arguments)
            print("Training data finished")
        preprocessed_dev_file = "../data/%s_%s_%s_dev_output.json" % (origin, language, tagging)
        if not os.path.isfile(preprocessed_dev_file):
            pkl_file = "../data/dev_%s_%s_%s_.pkl" % (origin, language, tagging)
            cache_arguments = ["python", join(XQA_PATH, "cache_dev.py"), origin, language, mode, pkl_file]
            print("Cache dev data: ", cache_arguments)
            subprocess.run(cache_arguments)
            dump_arguments = ["python", join(XQA_PATH, "dump_preprocessed_dev.py"), "--input_file", pkl_file,
                              "--output_dev_file", preprocessed_dev_file, "--language", language, "--tagging",
                              tagging]
            print("Dump dev data: ", dump_arguments)
            subprocess.run(dump_arguments)
            print("Dev data finished.")

        # run BERT
        vocab_file = config["BERT"]["VocabFile"]
        bert_config_file = config["BERT"]["BertConfigFile"]
        init_checkpoint = config["BERT"]["InitCheckpoint"]
        train_batch_size = config["BERT"]["TrainBatchSize"]
        predict_batch_size = config["BERT"]["PredictBatchSize"]
        num_gpus = config["BERT"]["NumGpus"]
        learning_rate = config["BERT"]["LearningRate"]
        epochs = config["BERT"]["Epochs"]
        max_seq_length = config["BERT"]["MaxSeqLength"]
        max_query_length = config["BERT"]["MaxQueryLength"]
        do_lower_case = config["BERT"]["DoLowerCase"]
        warm_up_portion = config["BERT"]["WarmUpProportion"]
        save_checkpoint_steps = config["BERT"]["SaveCheckPointSteps"]
        eval_steps = config["BERT"]["EvalSteps"]
        log_steps = config["BERT"]["LogSteps"]
        verbose_logging = config["BERT"]["VerboseLogging"]
        bert_arguments = ["python", join(XQA_PATH, "run_bert_open_qa_train.py"), "--vocab_file", vocab_file,
                          "--bert_config_file", bert_config_file, "--init_checkpoint", init_checkpoint,
                          "--train_file", preprocessed_train_file, "--eval_file", preprocessed_dev_file,
                          "--train_batch_size", train_batch_size, "--num_gpus", num_gpus, "--learning_rate", learning_rate,
                          "--num_train_epochs", epochs, "--max_seq_length", max_seq_length,
                          "--max_query_length", max_query_length, "--output_dir", new_directory,
                          "--do_lower_case", do_lower_case, "--warm_up_portion", warm_up_portion,
                          "--save_checkpoints_steps", save_checkpoint_steps, "--eval_steps", eval_steps,
                          "--log_steps", log_steps, "--verbose_logging", verbose_logging]
        print("Start training with bert: ", bert_arguments)
        subprocess.run(bert_arguments)
    print("Finished training at " + time.strftime('%Y%m%d-%H%M%S') + " after " + str(time.time() - start_time))
