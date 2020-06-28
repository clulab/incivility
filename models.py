import tensorflow as tf
import transformers


def from_transformer(transformer: transformers.TFPreTrainedModel,
                     n_outputs: int) -> tf.keras.Model:

    # Define inputs (token_ids, mask_ids, segment_ids)
    token_inputs = tf.keras.Input(shape=(None,), name='word_inputs', dtype='int32')
    mask_inputs = tf.keras.Input(shape=(None,), name='mask_inputs', dtype='int32')
    segment_inputs = tf.keras.Input(shape=(None,), name='segment_inputs', dtype='int32')

    # get contextualized token encodings from transformer
    token_encodings = transformer([token_inputs, mask_inputs, segment_inputs])[0]

    # get a sentence encoding from the token encodings
    sentence_encoding = tf.keras.layers.GlobalMaxPooling1D()(token_encodings)

    # Final output layer
    outputs = tf.keras.layers.Dense(n_outputs, activation='sigmoid', name='outputs')(sentence_encoding)

    # Define model
    return tf.keras.Model(inputs=[token_inputs, mask_inputs, segment_inputs], outputs=[outputs])
