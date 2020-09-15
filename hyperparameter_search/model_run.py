import numpy as np
import keras.backend as K
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

import os
import sys

from DeepDA_D2.DeepDA_hps.load_data import load_data

def r2(y_true, y_pred):
    SS_res = keras.backend.sum(keras.backend.square(y_true - y_pred), axis=0)
    SS_tot = keras.backend.sum(
        keras.backend.square(y_true - keras.backend.mean(y_true, axis=0)), axis=0
    )
    output_scores = 1 - SS_res / (SS_tot + keras.backend.epsilon())
    r2 = keras.backend.mean(output_scores)
    return r2


HISTORY = None


def run(config):
    global HISTORY
    (x_train, y_train), (x_valid, y_valid) = load_data()

    if config['activation'] == 'NA':
        config['activation'] = None

    model = Sequential()
    model.add(Dense(
        config['units'],
        activation=config['activation'],
        input_shape=tuple(np.shape(x_train)[1:])))
    model.add(Dense(1))

    model.summary()

    model.compile(loss='mse', optimizer=RMSprop(lr=config['lr']), metrics=[r2])

    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=1000,
                        verbose=1,
                        callbacks=[EarlyStopping(
                            monitor='val_r2',
                            mode='max',
                            verbose=1,
                            patience=10
                        )],
                        validation_data=(x_valid, y_valid))

    HISTORY = history.history

    return history.history['val_r2'][-1]


if __name__ == '__main__':
    config = {
        'units': 10,
        'activation': 'relu',
        'lr': 0.01
    }
    objective = run(config)
    print('objective: ', objective)
    import matplotlib.pyplot as plt
    plt.plot(HISTORY['val_r2'])
    plt.xlabel('Epochs')
    plt.ylabel('Objective: $R^2$')
    plt.grid()
    plt.show()