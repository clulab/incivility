import argparse
import os

import pandas as pd
import sklearn.model_selection as ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.csv_path, encoding="utf-8", errors="ignore") as csv_file:
        df = pd.read_csv(csv_file).dropna()
    [label_col] = [col for col in df.columns if "name" in col.lower()]

    train_df, dev_test_df = ms.train_test_split(
        df,
        train_size=args.train_frac,
        test_size=None,
        stratify=df[label_col],
        random_state=42)
    dev_df, test_df = ms.train_test_split(
        dev_test_df,
        train_size=0.5,
        test_size=None,
        stratify=dev_test_df[label_col],
        random_state=42)

    print(f"train: {len(train_df)}\n"
          f"dev:   {len(dev_df)}\n"
          f"test:  {len(test_df)}")

    csv_name, _ = os.path.splitext(args.csv_path)
    train_df.to_csv(csv_name + ".train.csv")
    dev_df.to_csv(csv_name + ".dev.csv")
    test_df.to_csv(csv_name + ".test.csv")