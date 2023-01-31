import argparse
import os
import pathlib
import pprint
import time

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

    training_args = dict(
        output_dir=model_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        report_to="wandb",
    )
    trainer_args = _get_trainer_args(
        hf_model_name,
        train_dirs,
        eval_dirs,
        eval_split="validation")

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
                    "min_iter": 8,
                    "eta": 2,
                }
            })


def test(model_dir: str, eval_dirs: list[str], split: str):
    trainer = transformers.Trainer(
        args=transformers.TrainingArguments(output_dir='temp', report_to="none"),
        **_get_trainer_args(model_dir, [], eval_dirs, eval_split=split))
    metrics = trainer.evaluate()
    time.sleep(1) # seems to be necessary in interactive job
    pprint.pprint(metrics)
    


def _get_trainer_args(
    pretrained_model_name,
    train_dirs: list[str],
    eval_dirs: list[str],
    eval_split: str):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name,
        model_max_length=512)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    def prepare_dataset(dataset_dir, split):
        return datasets.load_from_disk(dataset_dir)[split].map(
            lambda examples: {"label": examples["namecalling"]}
        ).map(
            lambda examples: tokenizer(examples["text"], truncation=True),
            batched=True
        )

    metrics = evaluate.combine(["f1", "precision", "recall"])
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return metrics.compute(
            predictions=np.argmax(logits, axis=1),
            references=labels,
            pos_label=1,
            average='binary')

    result = dict(
        model_init=lambda: transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=2,
            ignore_mismatched_sizes=True),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if train_dirs:
        result["train_dataset"] = datasets.concatenate_datasets(
            [prepare_dataset(d, "train") for d in train_dirs])
    if eval_dirs:
        result["eval_dataset"] = datasets.concatenate_datasets(
            [prepare_dataset(d, eval_split) for d in eval_dirs])
    return result


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
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("model_dir")
    test_parser.add_argument("--eval-dirs", nargs="+", metavar="dir", required=True)
    test_parser.add_argument("--split", default="test")
    test_parser.set_defaults(func=test)
    kwargs = vars(parser.parse_args())
    kwargs.pop("func")(**kwargs)
