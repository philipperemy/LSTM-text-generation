'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function

import random

import numpy as np
from keras.layers import Activation
from keras.layers import LSTM, Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of MAX_LEN characters
MAX_LEN = 40
STEP = 3
sentences_no_spaces = []
sentences = []
for i in range(0, len(text) - MAX_LEN, STEP):
    sentence = text[i: i + MAX_LEN]
    sentences_no_spaces.append(sentence.replace(' ', ''))
    sentences.append(sentence)

num_sentences = len(sentences_no_spaces)
print('nb sequences:', num_sentences)

print('Vectorization...')
X = np.zeros((num_sentences, MAX_LEN, len(chars)), dtype=np.bool)
Y = np.zeros((num_sentences, MAX_LEN, len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences_no_spaces):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        Y[i, t, char_indices[char]] = 1

c = int(0.75 * len(X))
X_train = X[:c]
Y_train = Y[:c]
S_X_train = sentences_no_spaces[:c]
S_Y_train = sentences[:c]

X_test = X[:c]
Y_test = Y[:c]
S_X_test = sentences_no_spaces[:c]
S_Y_test = sentences[:c]

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(MAX_LEN, len(chars)), return_sequences=True))
model.add(TimeDistributed(Dense(128)))
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
optimizer = Adam(lr=0.01, clipnorm=5.)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # model.fit(X_train, Y_train, batch_size=128, nb_epoch=1)
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=128, nb_epoch=1)

    for jj in range(50):
        test_index = random.randint(0, len(X_test))
        pred_probas = model.predict(np.expand_dims(X_test[test_index], axis=0), verbose=0)[0]
        pred_indexes = np.apply_along_axis(np.argmax, axis=1, arr=pred_probas)
        output_text = ''.join([indices_char[p] for p in pred_indexes])
        print('INPUT: ', S_X_test[test_index])
        print('OUTPUT: ', output_text)
        print('')

