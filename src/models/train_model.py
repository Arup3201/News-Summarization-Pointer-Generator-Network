## Project Folder Structure 👇

# Project-Folder
# -- data
# ---- external
# ---- interim
# ---- preprocessed
# ---- raw
# -- models
# ---- weights.best.hdf5
# -- notebooks
# ---- Text Summarization Project.ipynb
# ---- run model script and predict script.ipynb
# -- reports
# -- src
# ---- data
# ------ make_data.py
# ---- features
# ---- models
# ------ train_model.py
# ------ predict.py
# ---- visualizations

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Bidirectional, Dense, \
    AdditiveAttention, LayerNormalization, Add

BATCH_SIZE = 16  # From the paper

max_article_tokens = 400
max_summary_tokens = 100

article_vocab_size = 160000
summary_vocab_size = 50000


def load_data(path):
    data = pd.read_csv(path, header=None)
    data.columns = ['article', 'highlight']
    articles, summaries = data['article'].to_numpy(
    ), data['highlight'].to_numpy()

    return articles, summaries


def add_start_end(text):
    # If already any [START] or [END] token is present, remove them
    text = tf.strings.regex_replace(text, '[START]', '')
    text = tf.strings.regex_replace(text, '[END]', '')

    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

    return text


class Encoder(tf.keras.layers.Layer):
    def __init__(self, emb_out, lstm_units, text_processor):
        super(Encoder, self).__init__()
        self.emb_out = emb_out
        self.lstm_units = lstm_units
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()

        # Embedding layer will convert the tokens into vectors
        self.embedding = Embedding(self.vocab_size, self.emb_out)

        # Using Bidirectional layer with lstm cells
        self.rnn = Bidirectional(
            layer=LSTM(self.lstm_units,
                       # Return all output sequences
                       return_sequences=True),
            merge_mode='sum')

    def call(self, x):
        x = self.embedding(x)

        x = self.rnn(x)

        return x

    def convert_inputs(self, texts):
        texts = tf.convert_to_tensor(texts)

        if(len(texts.shape) == 0):
            texts = tf.convert_to_tensor(texts)[tf.newaxis]

        articles = self.text_processor(texts)
        enc_articles = self(articles)

        return enc_articles


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, use_scale=True, **kwargs):
        super(CrossAttention, self).__init__()

        self.additive_attention = AdditiveAttention(
            use_scale=use_scale, **kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()

    def call(self, x, context):
        attn_output, attn_scores = self.additive_attention(
            [x, context],
            return_attention_scores=True
        )

        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units, text_processor):
        super(Decoder, self).__init__()

        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(vocabulary=text_processor.get_vocabulary(),
                                                       mask_token='',
                                                       oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(vocabulary=text_processor.get_vocabulary(),
                                                       mask_token='',
                                                       oov_token='[UNK]',
                                                       invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        self.units = units

        self.embedding = Embedding(self.vocab_size, self.units, mask_zero=True)
        self.rnn = LSTM(self.units, return_sequences=True, return_state=True)
        self.attention = CrossAttention()
        self.output_layer = Dense(self.vocab_size)

    def call(self, context, x, state=None, return_state=False, return_attn=False):
        dec_in = self.embedding(x)

        x, h_state, m_state = self.rnn(dec_in, initial_state=state)

        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights

        logits = self.output_layer(x)

        if return_state:
            out = (logits, [h_state, m_state])

        else:
            out = logits

        return out


class Summarizer(tf.keras.Model):

    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, enc_emb_dim, units,
                 article_processor, summary_processor):

        super().__init__()

        self.encoder = Encoder(enc_emb_dim, units, article_processor)
        self.decoder = Decoder(units, summary_processor)

    def call(self, inputs):
        encoder_in, x = inputs
        context = self.encoder(encoder_in)
        logits = self.decoder(context, x)

        try:
            # Delete the keras mask, so keras doesn't scale the loss+accuracy.
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)


def main():
    args = sys.argv[1:]

    project_dir = args[0]

    train_file_path = os.path.join(
        project_dir, 'data/preprocessed/finished_files/final_train.csv')
    val_file_path = os.path.join(
        project_dir, 'data/preprocessed/finished_files/final_val.csv')

    train_articles_raw, train_summaries_raw = load_data(train_file_path)
    val_articles_raw, val_summaries_raw = load_data(val_file_path)

    BUFFER_SIZE = len(train_articles_raw)

    train_raw = tf.data.Dataset \
        .from_tensor_slices((train_articles_raw, train_summaries_raw)) \
        .shuffle(BUFFER_SIZE, seed=2023) \
        .batch(BATCH_SIZE)

    val_raw = tf.data.Dataset \
        .from_tensor_slices((val_articles_raw, val_summaries_raw)) \
        .shuffle(BUFFER_SIZE, seed=2023) \
        .batch(BATCH_SIZE)

    with open(os.path.join(project_dir, 'data/preprocessed/finished_files/vocab'), 'r') as reader:
        vocab = reader.read()

    vocab_data = [v.split(' ')[0] for v in vocab.split('\n')]
    vocab_data.remove('')
    vocab_data.insert(0, '[START]')
    vocab_data.insert(1, '[END]')

    article_processor = TextVectorization(max_tokens=article_vocab_size+2,  # 2 extra for '', [UNK]
                                          standardize=add_start_end,
                                          vocabulary=vocab_data[:article_vocab_size],
                                          output_sequence_length=max_article_tokens)

    summary_processor = TextVectorization(max_tokens=summary_vocab_size+2,  # 2 extra for '', [UNK]
                                          standardize=add_start_end,
                                          vocabulary=vocab_data[:summary_vocab_size],
                                          output_sequence_length=max_summary_tokens)

    def process_text(articles, summaries):
        encoder_in = article_processor(articles)
        target = summary_processor(summaries)
        target_in = target[:, :-1]
        target_out = target[:, 1:]

        return (encoder_in, target_in), target_out

    train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
    val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

    UNITS = 256
    EMB_DIM = 128

    model_dir = os.path.join(project_dir, "models")
    model_path = os.path.join(model_dir, "weights.best.hdf5")

    summarizer = Summarizer(
        EMB_DIM, UNITS, article_processor, summary_processor)

    adagrad_opt = tf.keras.optimizers.experimental.Adagrad(
        learning_rate=0.15,
        initial_accumulator_value=0.1,
        clipnorm=2,
    )

    summarizer.compile(optimizer=adagrad_opt,
                       loss=masked_loss,
                       metrics=[masked_acc, masked_loss])

    summarizer.evaluate(val_ds, steps=20, return_dict=True)

    summarizer.load_weights(model_path)

    history = summarizer.fit(
        train_ds.repeat(),
        epochs=30,
        steps_per_epoch=100,
        validation_data=val_ds,
        validation_steps=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="masked_acc", patience=3),

            tf.keras.callbacks.ModelCheckpoint(model_path, monitor="masked_acc", verbose=1,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='max')])

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch #')
    plt.ylabel('CE/token')
    plt.legend()

main()