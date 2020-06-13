import argparse

import pandas as pd
import tensorflow as tf
import transformers

import data
import models


tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
transformer = transformers.TFRobertaModel.from_pretrained('roberta-base')


def train(model_path, data_path, learning_rate, n_epochs):
    df = pd.read_csv(data_path, usecols=["text", "NAMECALLING"]).dropna()
    x = data.from_tokenizer(tokenizer, df["text"])
    y = df["NAMECALLING"].values

    model = models.from_transformer(transformer, 1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy('accuracy')])
    model.fit(x=x, y=y, epochs=n_epochs)
    model.save(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("model_path")
    train_parser.add_argument("data_path")
    train_parser.add_argument("--learning-rate", type=float, default=3e-5)
    train_parser.add_argument("--n-epochs", type=int, default=10)
    train_parser.set_defaults(func=train)
    args = parser.parse_args()

    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)
