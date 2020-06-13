import numpy as np


# converts texts into input matrices required by transformers
def from_tokenizer(tokenizer, texts, pad_token=0):
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

