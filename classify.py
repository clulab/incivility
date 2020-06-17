import argparse
import os
import subprocess
from typing import Text
import textwrap


def get_tokenizer():
    import transformers
    return transformers.RobertaTokenizer.from_pretrained('roberta-base')


def get_transformer():
    import transformers
    return transformers.TFRobertaModel.from_pretrained('roberta-base')


def train(model_path: Text,
          data_path: Text,
          n_rows: int,
          learning_rate: float,
          batch_size: int,
          n_epochs: int,
          qsub: bool,
          time: Text):

    if not qsub:
        if time is not None:
            raise ValueError("time limit not supported")

        import data
        import models
        import pandas as pd
        import tensorflow as tf

        df = pd.read_csv(data_path, nrows=n_rows, usecols=["text", "NAMECALLING"]).dropna()
        x = data.from_tokenizer(get_tokenizer(), df["text"])
        y = df["NAMECALLING"].values

        # set class weight inversely proportional to class counts
        counts = df["NAMECALLING"].value_counts()
        class_weight = (counts.max() / counts).to_dict()

        model = models.from_transformer(get_transformer(), 1)
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

    else:
        if time is None:
            raise ValueError("time limit required for qsub")
        model_prefix, _ = os.path.splitext(model_path)
        prefix = f"{model_prefix}.r{n_rows}.b{batch_size}.lr{learning_rate}"
        pbs_path = f"{prefix}.pbs"
        with open(pbs_path, "w") as pbs_file:
            pbs_file.write(textwrap.dedent(f"""
                #!/bin/bash

                #PBS -q windfall
                #PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
                #PBS -N {prefix}
                #PBS -W group_list=nlp
                #PBS -l walltime={time}

                module load singularity
                module load cuda10/10.1
                cd {os.path.dirname(os.path.realpath(__file__))}
                singularity exec --nv \\
                  $HOME/hpc-ml_centos7-python3.7-transformers2.11.sif \\
                  python3.7 classify.py train \\
                    --n-rows={n_rows} \\
                    --n-epochs={n_epochs} \\
                    --batch-size={batch_size} \\
                    --learning-rate={learning_rate} \\
                    {prefix}.model \\
                    {data_path}
                """))
        subprocess.run(["qsub", pbs_path])


def test(model_path: Text, data_path: Text, n_rows: int):
    import data
    import models
    import pandas as pd

    model = models.from_transformer(get_tokenizer(), 1)
    model.load_weights(model_path).expect_partial()

    df = pd.read_csv(data_path, nrows=n_rows, usecols=["text", "NAMECALLING"]).dropna()
    x = data.from_tokenizer(get_transformer(), df["text"])

    df.insert(1, "prediction", model.predict(x))
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("model_path")
    train_parser.add_argument("data_path")
    train_parser.add_argument("--qsub", action="store_true")
    train_parser.add_argument("--time")
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
