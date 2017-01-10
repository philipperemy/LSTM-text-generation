import random
import sys

import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from data_generator import read, VOCABULARY as chars

MAX_LEN = 40
STEP = 3


def sample(p, temperature=1.0):
    # helper function to sample an index from a probability array
    p = np.asarray(p).astype('float64')
    p = np.log(p) / temperature
    exp_p = np.exp(p)
    p = exp_p / np.sum(exp_p)
    probas = np.random.multinomial(1, p, 1)
    return np.argmax(probas)


def generate_sentences(model, text, char_indices, indices_char):
    start_index = random.randint(0, len(text) - MAX_LEN - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + MAX_LEN]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, MAX_LEN, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            predictions = model.predict(x, verbose=0)[0]
            next_index = int(sample(predictions, diversity))
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


def create_inputs(text, chars, char_indices):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - MAX_LEN, STEP):
        sentences.append(text[i: i + MAX_LEN])
        next_chars.append(text[i + MAX_LEN])
    print('nb sequences:', len(sentences))
    print('Vectorization...')
    x = np.zeros((len(sentences), MAX_LEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y


def main():
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    print(chars)
    voc_size = len(chars)
    print('total chars (vocabulary size):', voc_size)

    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, input_shape=(MAX_LEN, voc_size), return_sequences=True))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Dropout(0.2))
    model.add(Dense(voc_size))
    model.add(Activation('softmax'))

    optimizer = Adam(clipnorm=5., clipvalue=1.)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # train and validate the model, output generated text after each iteration
    for iteration in range(1, 600):
        print()
        print('-' * 50)
        print('Iteration', iteration)

        x_train, y_train = create_inputs(read(shuffle=True), chars, char_indices)
        model.fit(x_train, y_train, batch_size=512, nb_epoch=1, validation_split=0.2)

        generate_sentences(model, read(shuffle=True), char_indices, indices_char)


if __name__ == '__main__':
    main()
