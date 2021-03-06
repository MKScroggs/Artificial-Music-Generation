from keras.layers import LSTM, GRU, Dense, Activation, Dropout, SimpleRNN
from keras.models import Sequential, load_model
import keras.optimizers
from keras.callbacks import Callback
from keras import backend
from keras import backend as K
from keras.utils import np_utils
import numpy as np
import pandas
import math
from time import time

np.random.seed(6)


class LearningRateCallback(Callback):
    def __init__(self, update):
        super(LearningRateCallback, self).__init__()
        self.update = update

    def on_epoch_end(self, epoch, logs={}):
        new_lr = self.update(backend.get_value(self.model.optimizer.lr))
        backend.set_value(self.model.optimizer.lr, new_lr)


def view_network(model):
    for i, layer in enumerate(model.layers):
        config = layer.get_config()
        print("\nLayer: {}".format(i+1))
        for j, k in config.items():
            if j in ['name', 'input_dim', 'output_dim', 'activation', 'p']:
                print("{}: {}".format(j, k))
    print("")
    print("Optimizer: {}".format(type(model.optimizer).__name__))
    print("Loss: {}".format(model.loss))
    print("Current Learning Rate: {}".format
          (backend.get_value(model.optimizer.lr)))
    print("History Length: {}".format
          (model.layers[0].get_config().get("batch_input_shape")[1]))


'''
class LearningRateCallback(Callback):
    def __init__(self, update, verbose=False, monitor='loss',
                 patience=0, stop=3):
        super(LearningRateCallback, self).__init__()

        self.update = update
        self.monitor = monitor
        self.patience = patience

        self.wait = 0
        self.best = np.Inf
        self.stop = 3

        self.verbose = verbose

        self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):

            self.best = current
            self.wait = 0
        else:
            if self.verbose:
                print("\nCurrent: {}, Best: {}".format(current, self.best))
            if self.wait >= self.patience:
                """
                if self.wait >= self.stop:
                    self.model.stop_training = True
                    print("Stopping Early!")
                    return
                """
                new_lr = self.update(backend.get_value
                                    (self.model.optimizer.lr))
                backend.set_value(self.model.optimizer.lr, new_lr)
                if self.verbose:
                    print("\nNew Learning rate: {}".format(new_lr))
            self.wait += 1
'''


def get_SimpleRNN(shape, optimizer, loss, interval_width, history_length,
                  activation, dropout=0):
    model = Sequential()
    if len(shape) == 1:  # for a single layer network
            model.add(SimpleRNN(shape[0], input_shape=(history_length,
                                                       interval_width)))
            if dropout > 0:
                model.add(Dropout(dropout))
    else:  # for a multi-layer netowrk
        last_layer = len(shape) - 1
        for i, width in enumerate(shape):
            if i == 0:  # the first layer need to specify input_shape
                model.add(SimpleRNN(width,
                                    input_shape=(history_length,
                                                 interval_width),
                                    return_sequences=True))
            elif i == last_layer:  # the last layer does not return sequences
                model.add(SimpleRNN(width))
            else:  # middle layers need return sequences, but not input shape
                model.add(SimpleRNN(width, return_sequences=True))
            if dropout > 0:  # finally, if we are applying dropout, add it
                model.add(Dropout(dropout))
    # after all layers, add a dense layer and then the activation layer
    model.add(Dense(interval_width, activation=activation))

    model.compile(optimizer=optimizer, loss=loss)

    return model


def get_LSTM(shape, optimizer, loss, interval_width, history_length,
             activation, dropout=0, metrics=[]):
    model = Sequential()
    if len(shape) == 1:  # for a single layer network
            model.add(LSTM(shape[0],
                           input_shape=(history_length, interval_width),
                           consume_less="gpu"))
            if dropout > 0:
                model.add(Dropout(dropout))
    else:  # for a multi-layer netowrk
        last_layer = len(shape) - 1
        for i, width in enumerate(shape):
            if i == 0:  # the first layer need to specify input_shape
                model.add(LSTM(width,
                               input_shape=(history_length, interval_width),
                               consume_less="gpu", return_sequences=True))
            elif i == last_layer:  # the last layer does not return sequences
                model.add(LSTM(width, consume_less="gpu"))
            else:  # middle layers need return sequences, but not input shape
                model.add(LSTM(width, consume_less="gpu",
                               return_sequences=True))
            if dropout > 0:  # finally, if we are applying dropout, add it
                model.add(Dropout(dropout))

    # after all layers, add a dense layer and then the activation layer
    model.add(Dense(interval_width, activation=activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def get_GRU(shape, optimizer, loss, interval_width, history_length,
            activation, dropout=0):
    model = Sequential()
    if len(shape) == 1:  # for a single layer network
            model.add(GRU(shape[0],
                          input_shape=(history_length, interval_width),
                          consume_less="gpu"))
            if dropout > 0:
                model.add(Dropout(dropout))
    else:  # for a multi-layer netowrk
        last_layer = len(shape) - 1
        for i, width in enumerate(shape):
            if i == 0:  # the first layer need to specify input_shape
                model.add(GRU(width,
                              input_shape=(history_length, interval_width),
                              consume_less="gpu", return_sequences=True))
            elif i == last_layer:  # the last layer does not return sequences
                model.add(GRU(width, consume_less="gpu"))
            else:  # middle layers need return sequences, but not input shape
                model.add(GRU(width, consume_less="gpu",
                              return_sequences=True))
            if dropout > 0:  # finally, if we are applying dropout, add it
                model.add(Dropout(dropout))
        
    # after all layers, add a dense layer and then the activation layer
    model.add(Dense(interval_width, activation=activation))

    model.compile(optimizer=optimizer, loss=loss)

    return model


def get_network_from_file(filename):
    model = None
    try:
        model = load_model(filename + ".h5")
    except:  # in case the extension was specified
        model = load_model(filename)
    return model


def train_network(model, train_inputs, train_targets, val_inputs, val_targets,
                  epochs=1, callbacks=[], batch_size=512):
    model.fit(train_inputs, train_targets, nb_epoch=epochs,
              callbacks=callbacks, batch_size=batch_size,
              validation_data=(val_inputs, val_targets))
    return model


def test_melody_network(model, seed_sequence, sequence_length, interval_width,
                        history_length, temperature=1, count=8):
    generated_sequence = seed_sequence
    for i in range(sequence_length):
        # make a window into the generated sequence that is the proper length
        test_sequence = np.zeros((1, history_length, interval_width),
                                 dtype=np.bool)
        for j in range(history_length):
            for k in range(interval_width):
                test_sequence[0, j, k] = generated_sequence[0, j + i, k]

        prediction = model.predict(test_sequence)[0]

        # pick the best note
        predicted_matrix = sample(prediction, interval_width,
                                  temperature=temperature, count=count)
        generated_sequence = np.append(generated_sequence, predicted_matrix, 1)

    return generated_sequence


def test_accompaniment_network(model, seed_sequence, interval_width,
                               history_length, feature=1, count=3):
    generated_sequence = np.copy(seed_sequence)  # this is a full song's melody
    length = len(seed_sequence[0]) - history_length

    for i in range(length):
        # make a window into the generated sequence that is the proper length
        test_sequence = np.zeros((1, history_length, interval_width),
                                 dtype=np.bool)
        for j in range(history_length):
            for k in range(interval_width):
                test_sequence[0, j, k] = generated_sequence[0, j + i, k]

        # make a prediction
        prediction = model.predict(test_sequence)[0]

        # remove the seed note
        for n, note in enumerate(test_sequence[0][history_length-1]):
            # print (n, test_sequence[0][history_length-1][n], prediction[n])
            if note == 1:
                prediction[n] = 0

        # raw_input("")

        # pick the best note
        # predicted_matrix = get_above_percent(prediction,
        # interval_width, feature)
        predicted_matrix = sample(prediction, interval_width, feature, count)
        
        for n, note in enumerate(predicted_matrix[0][0]):
            if note == 1:
                generated_sequence[0][i + history_length - 1][n] = 1
                # add the predicted notes

    return generated_sequence


def get_top_n(prediction, interval_width, n=1):
    indeces = np.argpartition(prediction, -n)[-n:]
    predicted_matrix = np.zeros((1, 1, interval_width), dtype=np.bool)

    for index in indeces:
        predicted_matrix[0, 0, index] = True

    return predicted_matrix


def get_above_percent(prediction, interval_width, percent=.7):
    predicted_matrix = np.zeros((1, 1, interval_width), dtype=np.bool)

    for i, note in enumerate(prediction):
        if note >= percent:
            predicted_matrix[0, 0, i] = True

    return predicted_matrix


def sample(prediction, interval_width, temperature=0.5, count=8):
    #  reduce the input to the top 5 notes, to allow for some randomness,
    #  but prevent unwanted mulit-octave jumps
    indeces = np.argpartition(prediction, -5)[-5:]
    predicted_matrix = np.zeros(interval_width)
    for index in indeces:
        predicted_matrix[index] = prediction[index]
    #  helper function to sample an index from a probability array,
    #  from keras lstm example
    predictions = np.asarray(predicted_matrix).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    predicted_matrix = np.zeros((1, 1, interval_width), dtype=np.bool)
    predicted_matrix[0, 0, np.argmax(probabilities)] = 1
    return predicted_matrix
