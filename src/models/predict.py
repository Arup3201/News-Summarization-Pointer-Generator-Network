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
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Bidirectional, Dense, \
    AdditiveAttention, LayerNormalization, Add

BATCH_SIZE = 16  # From the paper

max_article_tokens = 400
max_summary_tokens = 100

article_vocab_size = 160000
summary_vocab_size = 50000

UNITS = 256
EMB_DIM = 128

args = sys.argv[1:]

project_dir = args[0]
txt_file_path = args[1]

with open(txt_file_path, 'r') as reader:
  contents = reader.read()

articles = contents.split('\n\n')

def main(articles):
    test_file_path = os.path.join(
        project_dir, 'data/preprocessed/finished_files/final_test.csv')
    test_articles_raw, test_summaries_raw = load_data(test_file_path)
    BUFFER_SIZE = len(test_articles_raw)

    test_raw = tf.data.Dataset \
        .from_tensor_slices((test_articles_raw, test_summaries_raw)) \
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

    test_ds = test_raw.map(process_text, tf.data.AUTOTUNE)


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

    summarizer.evaluate(test_ds, steps=20, return_dict=True)

    model_dir = os.path.join(project_dir, "models")
    model_path = os.path.join(model_dir, "weights.best.hdf5")
    summarizer.load_weights(model_path)

    print(summarizer.summarize(articles))


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


@Decoder.add_method
def get_initial_state(self, articles):
    batch_size = tf.shape(articles)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)


@Decoder.add_method
def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
    return result


@Decoder.add_method
def get_next_token(self, article, next_token, done, state):
    logits, state = self(
        article, next_token,
        state=state,
        return_state=True)

    next_token = tf.argmax(logits, axis=-1)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (next_token == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done, state


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


@Summarizer.add_method
def summarize(self,
              texts, *,
              max_length=120):
    # Process the input texts
    context = self.encoder.convert_inputs(texts)

    # Setup the loop inputs
    tokens = []
    next_token, done, state = self.decoder.get_initial_state(context)

    for _ in range(max_length):
        # Generate the next token
        next_token, done, state = self.decoder.get_next_token(
            context, next_token, done,  state)

        # Collect the generated tokens
        tokens.append(next_token)

    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)

    result = self.decoder.tokens_to_text(tokens)
    return result


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


main(articles)
