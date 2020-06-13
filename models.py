import tensorflow as tf
import transformers


def from_transformer(transformer: transformers.TFPreTrainedModel, n_outputs: int) -> tf.keras.Model:

    # Define inputs (token_ids, mask_ids, segment_ids)
    token_inputs = tf.keras.Input(shape=(None,), name='word_inputs', dtype='int32')
    mask_inputs = tf.keras.Input(shape=(None,), name='mask_inputs', dtype='int32')
    segment_inputs = tf.keras.Input(shape=(None,), name='segment_inputs', dtype='int32')

    # Load model and collect encodings
    token_encodings = transformer([token_inputs, mask_inputs, segment_inputs])[0]

    # Keep only [CLS] token encoding
    sentence_encoding = tf.squeeze(token_encodings[:, 0:1, :], axis=1)

    # Apply dropout
    sentence_encoding = tf.keras.layers.Dropout(0.1)(sentence_encoding)

    # Final output layer
    outputs = tf.keras.layers.Dense(n_outputs, activation='sigmoid', name='outputs')(sentence_encoding)

    # Define model
    return tf.keras.Model(inputs=[token_inputs, mask_inputs, segment_inputs], outputs=[outputs])
