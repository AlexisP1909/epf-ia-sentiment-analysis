from layers import SelfAttention
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense

import sys
# add parent directory to Python path for layers.py access
sys.path.append('..')
splits = {'train': 'data/train-00000-of-00001.parquet',
          'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
train_df = pd.read_parquet(
    "hf://datasets/google-research-datasets/poem_sentiment/" + splits["train"])
validation_df = pd.read_parquet(
    "hf://datasets/google-research-datasets/poem_sentiment/" + splits["validation"])
test_df = pd.read_parquet(
    "hf://datasets/google-research-datasets/poem_sentiment/" + splits["test"])
# verse_text, label
# label: 0 = negative, 1 = positive, 2 = no_impact, 3 = mixed


# Argument specification
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default=0,
                    help="Integer value representing a model configuration")
# CONFIG OPTIONS:
# 0: Simple Multi-Layer Perceptron Model
# 1: Simple Multi-Layer Perceptron Model w/ Self-Attention (Non-Penalized)
# 2: Simple Multi-Layer Perceptron Model w/ Self-Attention (Penalized)
args = parser.parse_args()
tf.random.set_seed(4000)
vocabulary_size = 10000  # choose 10k most-used words for truncated vocabulary
# choose 1000-word sequences, either pad or truncate sequences to this
sequence_length = 900
embedding_dims = 50      # number of dimensions to represent each word in vector space
batch_size = 100         # feed in the neural network in 100-example training batches
num_epochs = 100
config = 0
(X_train, Y_train), (X_validation,
                     Y_validation) = train_df['verse_text'].values, train_df['label'].values, validation_df['verse_text'].values, validation_df['label'].values

X_train = pad_sequences(sequences=X_train, maxlen=sequence_length)
X_validation = pad_sequences(sequences=X_validation, maxlen=sequence_length)

X = Input(shape=(sequence_length,), batch_size=batch_size)

embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_dims)(X)

# Optional Self-Attention Mechanisms
if config == 1:
    embedded, attention_weights = SelfAttention(size=50,
                                                num_hops=6,
                                                use_penalization=False)(embedded)
elif config == 2:
    embedded, attention_weights = SelfAttention(size=50,
                                                num_hops=6,
                                                use_penalization=True,
                                                penalty_coefficient=0.1)(embedded)

# Multi-Layer Perceptron
embedded_flattened = Flatten()(embedded)
fully_connected = Dense(units=250, activation='relu')(embedded_flattened)

# Prediction Layer
Y = Dense(units=1, activation='sigmoid')(fully_connected)

# Compile model
model = Model(inputs=X, outputs=Y)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train binary-classification model
model.fit(x=X_train, y=Y_train,
          validation_data=(X_validation, Y_validation),
          epochs=num_epochs, batch_size=batch_size)
y_pred = model.predict(test_df['verse_text'])
print(y_pred, test_df['label'])
