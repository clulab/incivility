import argparse
from typing import Text

import pandas as pd
import tensorflow as tf
import transformers

import data
import models


tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
transformer = transformers.TFRobertaModel.from_pretrained('roberta-base')


def train(model_path: Text,
          data_path: Text,
          n_rows: int,
          learning_rate: float,
          batch_size: int,
          n_epochs: int):
    df = pd.read_csv(data_path, nrows=n_rows, usecols=["text", "NAMECALLING"]).dropna()
    x = data.from_tokenizer(tokenizer, df["text"])
    y = df["NAMECALLING"].values

    # set class weight inversely proportional to class counts
    counts = df["NAMECALLING"].value_counts()
    class_weight = (counts.max() / counts).to_dict()

    model = models.from_transformer(transformer, 1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ])
    model.fit(x=x, y=y,
              epochs=n_epochs,
              batch_size=batch_size,
              class_weight=class_weight)
    model.save_weights(model_path)


def test(model_path: Text, data_path: Text, n_rows: int):
    model = models.from_transformer(transformer, 1)
    model.load_weights(model_path).expect_partial()

    df = pd.read_csv(data_path, nrows=n_rows, usecols=["text", "NAMECALLING"]).dropna()
    x = data.from_tokenizer(tokenizer, df["text"])

    df.insert(1, "prediction", model.predict(x))
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("model_path")
    train_parser.add_argument("data_path")
    train_parser.add_argument("--n-rows", type=int)
    train_parser.add_argument("--learning-rate", type=float, default=3e-5)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--n-epochs", type=int, default=10)
    train_parser.set_defaults(func=train)
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("model_path")
    test_parser.add_argument("data_path")
    test_parser.add_argument("--n-rows", type=int)
    test_parser.set_defaults(func=test)
    args = parser.parse_args()

    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)
