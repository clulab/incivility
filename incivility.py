import dataclasses

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

    def load(self, split='train'):
        df = pd.read_csv(
            self.filename_format.format(split),
            usecols=[self.text_column, self.namecalling_column],
            converters={self.namecalling_column: to_binary_label})
        df = df.rename(columns={
            self.text_column: "text",
            self.namecalling_column: "namecalling"})
        df = df[["namecalling", "text"]]
        return datasets.Dataset.from_pandas(df, split=split)


ARIZONA_DAILY_STAR = IncivilityData(
    "data/ADS/{}_data_with_tag_and_aux.csv",
    "text",
    "NAMECALLING")

PRIMARIES_2020 = IncivilityData(
    "data/Primaries2020/Consolidated Intercoder Data Tweets pre 2020 non-quotes removed_utf8.{}.csv",
    "Tweettext",
    "NameCalling")

RUSSIAN_TROLLS = IncivilityData(
    "data/Troll/Troll Data Annotated.{}.csv",
    "Tweet",
    "Name calling (1 = y; 0 = n)")

TUCSON = IncivilityData(
    "data/Tucson/Tucson Annotation Final Round Merged.{}.csv",
    "Tweet",
    "NAME CALLING (Yes= 1; No = 0).x")


def train(model_name: str, data: IncivilityData):
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
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

    data_train = data.load('train').map(set_label).map(tokenize, batched=True)
    data_dev = data.load('dev').map(set_label).map(tokenize, batched=True)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2)

    training_args = transformers.TrainingArguments(
        output_dir=model_name,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
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
    train("tucson_model", TUCSON)
