import numpy as np
import pandas as pd
import transformers
from typing import Mapping, Sequence, Text, Union


# converts texts into input matrices required by transformers
def from_tokenizer(tokenizer: transformers.PreTrainedTokenizer,
                   texts: Sequence[Text],
                   pad_token: int = 0) -> Mapping[Text, np.ndarray]:
    rows = [tokenizer.encode(text,
                             add_special_tokens=True,
                             max_length=tokenizer.model_max_length)
            for text in texts]
    shape = (len(rows), max(len(row) for row in rows))
    token_ids = np.full(shape=shape, fill_value=pad_token)
    is_token = np.zeros(shape=shape)
    for i, row in enumerate(rows):
        token_ids[i, :len(row)] = row
        is_token[i, :len(row)] = 1
    return dict(
        word_inputs=token_ids,
        mask_inputs=is_token,
        segment_inputs=np.zeros(shape=shape))


def read_ads_csv(
        data_path: Text,
        n_rows: Union[int, None],
        tokenizer: transformers.PreTrainedTokenizer) \
        -> (pd.DataFrame, np.ndarray, np.ndarray):
    df = pd.read_csv(data_path,
                     nrows=n_rows,
                     usecols=["text", "NAMECALLING"]).dropna()
    x = from_tokenizer(tokenizer, df["text"])
    y = df["NAMECALLING"].values
    return df, x, y
