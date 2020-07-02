import argparse
import os
import subprocess
from typing import Sequence, Text
import textwrap

import sklearn
import tensorflow as tf
import tensorflow_addons as tfa
import transformers

import data
import models
import ga


def train(model_path: Text,
          train_data_path: Text,
          dev_data_path: Text,
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
        train_df, train_x, train_y = data.read_ads_csv(
            data_path=train_data_path,
            n_rows=n_rows,
            tokenizer=tokenizer)
        _, dev_x, dev_y = data.read_ads_csv(
            data_path=dev_data_path,
            n_rows=n_rows,
            tokenizer=tokenizer)

        # set class weight inversely proportional to class counts
        counts = train_df["NAMECALLING"].value_counts()
        class_weight = (counts.max() / counts).to_dict()

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
                    {train_data_path} \\
                    {dev_data_path}
                """))
        subprocess.run(["qsub", pbs_path])


def test(model_paths: Sequence[Text],
         data_path: Text,
         pretrained_model_name: Text,
         n_rows: int,
         batch_size: int,
         verbose: bool):

    width = max(len(p) for p in model_paths)
    headers = ["precision", "recall", "f1-score", "support"]
    header = (f'{{:<{width}s}} ' + ' {:>9}' * 4).format('', *headers)
    row_fmt = f'{{:<{width}s}} ' + ' {:>9.3f}' * 3 + ' {:>9}'

    rows = []
    for model_path in model_paths:
        tf.keras.backend.clear_session()

        # load the transformer model
        model_for = transformers.TFRobertaModel.from_pretrained
        model = models.from_transformer(
            transformer=model_for(pretrained_model_name),
            n_outputs=1)
        model.load_weights(model_path).expect_partial()

        # load the tokenizer and tokenize the test data
        tokenizer_for = transformers.AutoTokenizer.from_pretrained
        df, x, y_ref = data.read_ads_csv(
            data_path=data_path,
            n_rows=n_rows,
            tokenizer=tokenizer_for(pretrained_model_name))

        # predict on the test data
        y_pred_scores = model.predict(x, batch_size=batch_size)
        y_pred = (y_pred_scores >= 0.5).astype(int).ravel()

        # evaluate predictions
        stats_arrays = sklearn.metrics.precision_recall_fscore_support(
            y_ref, y_pred, labels=[1])
        stats = [a.item() for a in stats_arrays]
        row = [model_path] + stats
        rows.append(row)

        # if requested, print detailed results for this model
        if verbose:
            print("=" * len(header))
            print(header)
            print(row_fmt.format(*row))
            print("=" * len(header))
            df.insert(1, "prediction", y_pred_scores)
            print(df)
            print()

    # print results for all models
    print(header)
    for row in rows:
        print(row_fmt.format(*row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-name", default="roberta-base")
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("model_path")
    train_parser.add_argument("train_data_path")
    train_parser.add_argument("dev_data_path", nargs='?')
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
    test_parser.add_argument("data_path")
    test_parser.add_argument("--n-rows", type=int)
    test_parser.add_argument("--batch-size", type=int, default=1)
    test_parser.add_argument("--verbose", action="store_true")
    test_parser.set_defaults(func=test)
    args = parser.parse_args()

    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)
