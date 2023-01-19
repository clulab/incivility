import argparse
import pathlib
import os

import numpy as np
import pandas as pd

import datasets
import evaluate
import transformers


def to_binary_label(s: str):
    if s == 'o':
        return 0
    f = float(s)
    if f.is_integer():
        return int(f)
    raise ValueError


def create_datasets(data_dir: str, output_dir: str):
    data_path = pathlib.Path(data_dir)
    output_path = pathlib.Path(output_dir)
    info = {
        "arizona_daily_star_comments": (
            "{split}_data_with_tag_and_aux.csv",
            "text",
            "NAMECALLING"),
        "us_presidential_primary_tweets": (
            "Consolidated Intercoder Data Tweets pre 2020 non-quotes "
            "removed_utf8.{split}.csv",
            "Tweettext",
            "NameCalling"),
        "russian_troll_tweets": (
            "Troll Data Annotated.{split}.csv",
            "Tweet",
            "Name calling (1 = y; 0 = n)"),
        "tucson_official_tweets": (
            "Tucson Annotation Final Round Merged.{split}.csv",
            "Tweet",
            "NAME CALLING (Yes= 1; No = 0).x"),
    }

    for d in data_path.iterdir():
        if d.is_dir():
            name = d.name.lower().replace("-", "_")
            filename_format, text_column, namecalling_column = info[name]
            dataset_dict = {}
            for split, hf_split in [("train", datasets.Split.TRAIN),
                                    ("dev", datasets.Split.VALIDATION),
                                    ("test", datasets.Split.TEST)]:
                df = pd.read_csv(
                    d / filename_format.format(split=split),
                    usecols=[text_column, namecalling_column],
                    converters={namecalling_column: to_binary_label})
                df = df.rename(columns={
                    text_column: "text",
                    namecalling_column: "namecalling"})
                df = df[["namecalling", "text"]]
                dataset = datasets.Dataset.from_pandas(df.dropna())
                dataset_dict[str(hf_split)] = dataset
            datasets.DatasetDict(dataset_dict).save_to_disk(
                str(output_path / f"incivility_{name}"))


def train(hf_model_name: str,
          model_dir: str,
          train_dirs: list[str],
          eval_dirs: list[str],
          param_search: bool):
    model_name = pathlib.Path(model_dir).name

    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1_metric.compute(predictions=predictions, references=labels,
                                 pos_label=1, average='binary')

    def set_label(examples):
        return {"label": examples["namecalling"]}

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True)
    
    def prepare_dataset(dataset_dir, split):
        dataset = datasets.load_from_disk(dataset_dir)[split]
        return dataset.map(set_label).map(tokenize, batched=True)

    train_datasets = [prepare_dataset(d, "train") for d in train_dirs]
    eval_datasets = [prepare_dataset(d, "validation") for d in eval_dirs]

    def model_init():
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,
            num_labels=2)

    training_args = dict(
        output_dir=model_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        report_to="wandb",
    )
    trainer_args = dict(
        model_init=model_init,
        train_dataset=datasets.concatenate_datasets(train_datasets),
        eval_dataset=datasets.concatenate_datasets(eval_datasets),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    os.environ["WANDB_PROJECT"] = "incivility"

    if not param_search:
        args = transformers.TrainingArguments(
            learning_rate=5e-5,
            run_name=model_name,
            **training_args)
        trainer = transformers.Trainer(args=args, **trainer_args)
        trainer.train()
    
    else:
        args = transformers.TrainingArguments(**training_args)
        trainer = transformers.Trainer(args=args, **trainer_args)
        trainer.hyperparameter_search(
            direction="maximize", 
            backend="wandb", 
            n_trials=32,
            hp_space=lambda trial: {
                "method": "random",
                "name": f"sweep_{model_name}",
                "metric": {
                    "name": "eval_f1",
                    "goal": "maximize",
                },
                "parameters": {
                    "learning_rate": {
                        "distribution": "uniform",
                        "min": 1e-6,
                        "max": 1e-4},
                    "seed": {
                        "distribution": "int_uniform",
                        "min": 1,
                        "max": 8},
                },
                "early_terminate": {
                    "type": "hyperband",
                    "min_iter": 2,
                    "eta": 2,
                }
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument("data_dir")
    dataset_parser.add_argument("output_dir")
    dataset_parser.set_defaults(func=create_datasets)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("model_dir")
    train_parser.add_argument("--train-dirs", nargs="+", metavar="dir", required=True)
    train_parser.add_argument("--eval-dirs", nargs="+", metavar="dir", required=True)
    train_parser.add_argument("--param-search", action="store_true")
    train_parser.add_argument("--base-model", dest="hf_model_name", default="roberta-base")
    train_parser.set_defaults(func=train)
    kwargs = vars(parser.parse_args())
    kwargs.pop("func")(**kwargs)
