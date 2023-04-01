import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import re
import string
import random
from transformers import AutoTokenizer
import pandas as pd
import cProfile
import pstats
import multiprocessing
import sqlite3


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", add_special_tokens=False)
tokens = ['[START]', '[answer]', '[END]']
tokenizer.add_special_tokens({'additional_special_tokens': tokens})

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions




vocab_size = 40000  # Only consider the top 30k words
maxlen = 100  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer


def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    adam = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)

    model.compile(
        'adam', loss=[loss_fn, None], metrics=['accuracy']
    )  # No loss and optimization based on word embeddings from transformer block
    return model

vocab = tokenizer.get_vocab()
subword_list = [None] * len(vocab)
for subword, index in vocab.items():
    subword_list[index] = subword
vocab = subword_list

print('loaded vocab')

class generate_text():

    x = 'suck my dikc'

    def __init__(self, model, max_tokens, start_tokens, index_to_word, top_k=20, print_every=1):
        self.model = model
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k
    
    def sample_from(self, logits):
        logits /= 1.0
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)
    
    def detokenize(self, number):
        return self.index_to_word[number]
    
    def generate(self):
        tokenixer = AutoTokenizer.from_pretrained("bert-base-uncased", max_vocab_size=vocab_size)
        
        start_tokens = [_ for _ in self.start_tokens]

        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[-maxlen:]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            if sample_token == 102:
                break
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = "".join(
            tokenizer.decode(self.start_tokens + tokens_generated)
        )
        print(f"generated text:\n{txt}\n")
        return txt

model = create_model()
model.load_weights('Transformer/transformer.h5')


def clean_msg(msg):
    index = msg.index("[answer]")
    clean_msg = msg[index+len("[answer]"):].strip() 

    return clean_msg

def response_msg(msg):
    msg = "[START] " + msg + " [answer]"
    start_tokens = tokenizer.encode(msg, add_special_tokens=False)
    max_tokens = 160
    text_gen = generate_text(model, max_tokens, start_tokens, vocab)
    response = text_gen.generate()

    response = clean_msg(response)
    return response