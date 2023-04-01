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

from transformers import AutoTokenizer, BertTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", add_special_tokens=False)
tokens = ['[START]', '[answer]', '[END]']
tokenizer.add_special_tokens({'additional_special_tokens': tokens})

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class dataset_gen():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", add_special_tokens=False)
        tokens = ['[START]', '[answer]', '[END]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': tokens})

    def split_data(self, data):
        data_chunks = []
        for i in range(0, len(data) - maxlen, 512):
            data_chunks.append(data[i:i+512])

        return data_chunks

    def encode_chunk(self, chunk):

        encoded = self.tokenizer.encode(chunk, max_length=512, truncation=True, add_special_tokens=False)

        return encoded

    def encode_data(self, data):
        with multiprocessing.Pool() as pool:
            encoded_chunks = pool.map(self.encode_chunk, data)
        
        return [encoded for chunk in encoded_chunks for encoded in chunk]

    def save_data(self, data):
        conn = sqlite3.connect('Transformer/data.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE random_numbers
                    (id INTEGER PRIMARY KEY,
                    number INTEGER)''')
        for num in data:
            c.execute(f"INSERT INTO random_numbers (number) VALUES ({num})")
        conn.commit()
        conn.close()

    def extract_from_csv(self):

        df = pd.read_csv('StackExchange/qna.csv')

        joined_data = ''
        for index, row in df.iterrows():
            joined_data += '[START] ' + row['question'] + '\n' + row['contents'] + " [answer] " + row['answer_contents'] + '\n [END]'

        return joined_data

    def dataset(self):

        df = pd.read_csv('WikipediaScraper/data.csv')


        columns = df['article'].tolist()
        new_columns = list(map(lambda x: "[START] " + str(x) + " [END]", columns))
        joined_data = '\n'.join(new_columns)
        
        joined_data = joined_data + '\n' + self.extract_from_csv()

        splited = self.split_data(joined_data)

        encoded = self.encode_data(splited)
        encoded_chunks = encoded

        vocab = tokenizer.get_vocab()
        subword_list = [None] * len(vocab)
        for subword, index in vocab.items():
            subword_list[index] = subword
        vocab = subword_list

        #self.save_data(encoded_chunks)

        return df, vocab, encoded_chunks, joined_data, self.tokenizer
    
gen = dataset_gen()
df, vocab, encoded_chunks, joined_data, gen_tokenizer = gen.dataset()

class DataGenerator():

    def __init__(self,batch_size, maxlen, threadsafe=True, vocab_size=40000):

        self.batch_size = batch_size
        self.maxlen = maxlen
        
        if threadsafe:
            self.conn = sqlite3.connect('Transformer/data.db', check_same_thread=False)
        else:
            self.conn = sqlite3.connect('Transformer/data.db')
        cursor = self.conn.execute("SELECT COUNT(*) FROM random_numbers")
        self.count = cursor.fetchone()[0]
        print(self.count)

    def random_index(self):
        return random.randint(1, self.count - self.maxlen - 1)

    def get_data(self):
        c = self.conn.cursor()
        index = self.random_index()
        c.execute("SELECT * FROM random_numbers WHERE id BETWEEN ? AND ?", (index, index + self.maxlen+1))
        sequence_data = [row[1] for row in c]
        return sequence_data
    
    def label_data(self, data):

        #data = tf.cast(data, dtype=tf.int32)
        #data = tf.convert_to_tensor(data, dtype=tf.int32)
        x = data[:-1]
        y = data[1:]
        return x, y

    def generate(self):
        while True:
            sequence = []
            x, y = [], []
            x_list, y_list = [], []
            for _ in range(self.batch_size):
                sequence = self.get_data()
                x, y = self.label_data(sequence)
                x_list.append(x)
                y_list.append(y)

            if len(x_list) == self.batch_size:
                try:
                    x_list = np.array(x_list)
                    y_list = np.array(y_list)
                    yield x_list, y_list
                except ValueError:
                    print(x_list)
                    continue

from keras.callbacks import ModelCheckpoint

callbacks = {
    'checkpoint': ModelCheckpoint('Transformer/model-{epoch:03d}.h5', monitor='loss', verbose=0, save_best_only=True, mode='auto'),
}

data_generator = DataGenerator(128, maxlen, threadsafe=True, vocab_size=40000)
data_gen = data_generator.generate()

tf.debugging.disable_traceback_filtering()

def train():
    model = create_model()

    steps_size = 1000

    if os.path.exists('Transformer/model-006.h5'):
        model.load_weights('Transformer/model-006.h5')

    model.fit(data_gen, steps_per_epoch = steps_size, verbose=1, epochs=10, callbacks=[callbacks['checkpoint']])

    model.save_weights('Transformer/transformer.h5')

    return model

def main():
    with cProfile.Profile() as pr:
        model = train()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats('transformer.prof')
    return model

model = main()