import argparse
import os
import subprocess
from typing import Sequence, Text
import textwrap

import numpy as np
import sklearn
import tensorflow as tf
import tensorflow_addons as tfa
import transformers

import data
import models
import ga


def train(model_path: Text,
          train_data_paths: Sequence[Text],
          dev_data_paths: Sequence[Text],
          pretrained_model_name: Text,
          n_rows: int,
          learning_rate: float,
          batch_size: int,
          grad_accum_steps: int,
          n_epochs: int,
          qsub: bool,
          time: Text):

    if not qsub:
        if time is not None:
            raise ValueError("time limit not supported")

        tokenizer_for = transformers.AutoTokenizer.from_pretrained
        tokenizer = tokenizer_for(pretrained_model_name)
        train_x, train_y = data.read_namecalling_csvs_to_xy(
            data_paths=train_data_paths,
            n_rows=n_rows,
            tokenizer=tokenizer)
        dev_x, dev_y = data.read_namecalling_csvs_to_xy(
            data_paths=dev_data_paths,
            n_rows=n_rows,
            tokenizer=tokenizer)

        # set class weight inversely proportional to class counts
        counts = np.bincount(train_y)
        class_weight = dict(enumerate(counts.max() / counts))

        # determine optimizer
        optimizer_kwargs = dict(
            learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
        if grad_accum_steps != 1:
            optimizer_class = ga.AdamGA
            optimizer_kwargs.update(grad_accum_steps=grad_accum_steps)
        else:
            optimizer_class = tf.optimizers.Adam

        model_for = transformers.TFRobertaModel.from_pretrained
        model = models.from_transformer(
            transformer=model_for(pretrained_model_name),
            n_outputs=1)
        model.compile(
            optimizer=optimizer_class(**optimizer_kwargs),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tfa.metrics.F1Score(num_classes=1, threshold=0.5),
            ])
        model.fit(x=train_x, y=train_y,
                  validation_data=(dev_x, dev_y),
                  epochs=n_epochs,
                  batch_size=batch_size,
                  class_weight=class_weight,
                  callbacks=tf.keras.callbacks.ModelCheckpoint(
                      filepath=model_path,
                      monitor="val_f1_score",
                      mode="max",
                      verbose=1,
                      save_weights_only=True,
                      save_best_only=True))

    else:
        if time is None:
            raise ValueError("time limit required for qsub")
        model_prefix, _ = os.path.splitext(model_path)
        n_rows_str = "all" if n_rows is None else n_rows
        prefix = f"{model_prefix}.{pretrained_model_name}.r{n_rows_str}.b{batch_size}.ga{grad_accum_steps}.lr{learning_rate}"
        pbs_path = f"{prefix}.pbs"

        def format_paths(paths):
            return ' '.join(f'"{p}"' for p in paths)

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
                    {'' if n_rows is None else f'--n-rows={n_rows}'} \\
                    --n-epochs={n_epochs} \\
                    --batch-size={batch_size} \\
                    --grad-accum-steps={grad_accum_steps} \\
                    --learning-rate={learning_rate} \\
                    {prefix}.model \\
                    --train-data {format_paths(train_data_paths)} \\
                    --dev-data {format_paths(dev_data_paths)}
                """))
        subprocess.run(["qsub", pbs_path])


def test(model_paths: Sequence[Text],
         test_data_paths: Sequence[Text],
         pretrained_model_name: Text,
         n_rows: int,
         batch_size: int,
         verbose: bool):

    width = max(len(p) for p in model_paths + test_data_paths)
    headers = ["precision", "recall", "f1-score", "support"]
    header_fmt = f'{{:<{width}s}} ' + ' {:>9}' * 4
    row_fmt = f'{{:<{width}s}} ' + ' {:>9.3f}' * 3 + ' {:>9}'

    # load the tokenizer model
    tokenizer_for = transformers.AutoTokenizer.from_pretrained
    tokenizer = tokenizer_for(pretrained_model_name)

    # load the pre-trained transformer model
    model_for = transformers.TFRobertaModel.from_pretrained
    transformer = model_for(pretrained_model_name)

    test_data_rows = {p: [] for p in test_data_paths}
    for model_path in model_paths:
        tf.keras.backend.clear_session()

        # load the fine-tuned transformer model
        model = models.from_transformer(transformer=transformer, n_outputs=1)
        model.load_weights(model_path).expect_partial()

        for data_path in test_data_paths:

            # tokenize the test data
            df = data.read_namecalling_csv(data_path=data_path, n_rows=n_rows)
            x, y_ref = data.namecalling_df_to_xy(df=df, tokenizer=tokenizer)

            # predict on the test data
            y_pred_scores = model.predict(x, batch_size=batch_size)
            y_pred = (y_pred_scores >= 0.5).astype(int).ravel()

            # evaluate predictions
            stats_arrays = sklearn.metrics.precision_recall_fscore_support(
                y_ref, y_pred, labels=[1])
            stats = [a.item() for a in stats_arrays]
            row = [model_path] + stats
            test_data_rows[data_path].append(row_fmt.format(*row))

            # if requested, print detailed results for this model
            if verbose:
                header = header_fmt.format(data_path, *headers)
                print("=" * len(header))
                print(header)
                print(row_fmt.format(*row))
                print("=" * len(header))
                df.insert(1, "prediction", y_pred_scores)
                print(df)
                print()

    # print results for all models on all datasets
    for data_path, rows in test_data_rows.items():
        print(header_fmt.format(data_path, *headers))
        for row in rows:
            print(row)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-name", default="roberta-base")
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("model_path")
    train_parser.add_argument("--train-data", dest="train_data_paths", nargs='+',
                              metavar="PATH", required=True)
    train_parser.add_argument("--dev-data", dest="dev_data_paths", nargs='+',
                              metavar="PATH", required=True)
    train_parser.add_argument("--qsub", action="store_true")
    train_parser.add_argument("--time")
    train_parser.add_argument("--n-rows", type=int)
    train_parser.add_argument("--learning-rate", type=float, default=3e-5)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_parser.add_argument("--n-epochs", type=int, default=10)
    train_parser.set_defaults(func=train)
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("model_paths", nargs="+", metavar="model_path")
    test_parser.add_argument("--test-data", dest="test_data_paths", nargs='+',
                             metavar="PATH", required=True)
    test_parser.add_argument("--n-rows", type=int)
    test_parser.add_argument("--batch-size", type=int, default=1)
    test_parser.add_argument("--verbose", action="store_true")
    test_parser.set_defaults(func=test)
    args = parser.parse_args()

    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)
