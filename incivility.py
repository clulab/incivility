import argparse
import dataclasses
import pathlib

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


@dataclasses.dataclass
class IncivilityData:
    filename_format: str
    text_column: str
    namecalling_column: str

    def load(self, data_dir, split='train'):
        df = pd.read_csv(
            self.filename_format.format(data_dir=data_dir, split=split),
            usecols=[self.text_column, self.namecalling_column],
            converters={self.namecalling_column: to_binary_label})
        df = df.rename(columns={
            self.text_column: "text",
            self.namecalling_column: "namecalling"})
        df = df[["namecalling", "text"]]
        return datasets.Dataset.from_pandas(df, split=split)

DATA = {
    "ADS": IncivilityData(
        "{data_dir}/{split}_data_with_tag_and_aux.csv",
        "text",
        "NAMECALLING"),
    "Primaries2020": IncivilityData(
        "{data_dir}/Consolidated Intercoder Data Tweets pre 2020 non-quotes removed_utf8.{split}.csv",
        "Tweettext",
        "NameCalling"),
    "Troll": IncivilityData(
        "{data_dir}/Troll Data Annotated.{split}.csv",
        "Tweet",
        "Name calling (1 = y; 0 = n)"),
    "Tucson": IncivilityData(
        "{data_dir}/Tucson Annotation Final Round Merged.{split}.csv",
        "Tweet",
        "NAME CALLING (Yes= 1; No = 0).x"),
}


def train(data_dir: str, model_name: str):
    hf_model_name = "roberta-base"
    data = DATA[pathlib.PurePath(data_dir).name]

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

    data_train = data.load(data_dir, 'train').map(set_label).map(tokenize, batched=True)
    data_dev = data.load(data_dir, 'dev').map(set_label).map(tokenize, batched=True)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        hf_model_name,
        num_labels=2)

    training_args = transformers.TrainingArguments(
        output_dir=model_name,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1"
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("model_name")
    train(**vars(parser.parse_args()))
