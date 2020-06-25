import argparse
import os
import subprocess
from typing import Text
import textwrap


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

        import data
        import models
        import ga
        import tensorflow as tf
        import transformers

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
            ])
        model.fit(x=train_x, y=train_y,
                  validation_data=(dev_x, dev_y),
                  epochs=n_epochs,
                  batch_size=batch_size,
                  class_weight=class_weight)
        model.save_weights(model_path)

    else:
        if time is None:
            raise ValueError("time limit required for qsub")
        model_prefix, _ = os.path.splitext(model_path)
        n_rows_str = "all" if n_rows is None else n_rows
        prefix = f"{model_prefix}.r{n_rows_str}.b{batch_size}.ga{grad_accum_steps}.lr{learning_rate}"
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


def test(model_path: Text,
         data_path: Text,
         pretrained_model_name: Text,
         n_rows: int):
    import data
    import models
    import transformers

    model_for = transformers.TFRobertaModel.from_pretrained
    model = models.from_transformer(
        transformer=model_for(pretrained_model_name),
        n_outputs=1)
    model.load_weights(model_path).expect_partial()

    tokenizer_for = transformers.AutoTokenizer.from_pretrained
    df, x, _ = data.read_ads_csv(
        data_path=data_path,
        n_rows=n_rows,
        tokenizer=tokenizer_for(pretrained_model_name))

    df.insert(1, "prediction", model.predict(x))
    print(df)


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
    test_parser.add_argument("model_path")
    test_parser.add_argument("data_path")
    test_parser.add_argument("--n-rows", type=int)
    test_parser.set_defaults(func=test)
    args = parser.parse_args()

    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)
