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
    def __init__(self, update, verbose=False, monitor='loss', patience=0, stop=3):
        super(LearningRateCallback, self).__init__()
        
        self.update = update
        self.monitor = monitor
        self.patience = patience

        self.wait = 0
        self.best = np.Inf
        self.stop = 3

        self.verbose = verbose

        self.monitor_op = np.less

    #def on_train_begin(self, logs={}):
        #self.wait = 0       # Allow instances to be re-used
        #self.best = np.Inf if self.monitor_op == np.less else -np.Inf

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
                new_lr = self.update(backend.get_value(self.model.optimizer.lr))
                backend.set_value(self.model.optimizer.lr, new_lr)
                if self.verbose:
                    print("\nNew Learning rate: {}".format(new_lr)) 
            self.wait += 1

def get_SimpleRNN(shape, optimizer, loss, interval_width, history_length, activation, dropout=0):
    model = Sequential()
    if len(shape) == 1: # for a single layer network
            model.add(SimpleRNN(shape[0], input_shape=(history_length, interval_width)))
            if dropout > 0:
                model.add(Dropout(dropout))
    else: # for a multi-layer netowrk
        last_layer = len(shape) - 1
        for i, width in enumerate(shape):
            if i == 0: # the first layer need to specify input_shape
                model.add(SimpleRNN(width, input_shape=(history_length, interval_width), return_sequences=True))
            elif i == last_layer: #the last layer does not return sequences
                model.add(SimpleRNN(width))
            else: # middle layers need return sequences, but not input shape
                model.add(SimpleRNN(width, return_sequences=True))
           
            if dropout > 0: # finally, if we are applying dropout, add it
                model.add(Dropout(dropout))
        
    #after all layers, add a dense layer and then the activation layer
    model.add(Dense(interval_width, activation=activation))

    model.compile(optimizer=optimizer, loss=loss)

    return model


def get_LSTM(shape, optimizer, loss, interval_width, history_length, activation, dropout=0, metrics=[]):
    model = Sequential()
    if len(shape) == 1: # for a single layer network
            model.add(LSTM(shape[0], input_shape=(history_length, interval_width), consume_less="gpu"))
            if dropout > 0:
                model.add(Dropout(dropout))
    else: # for a multi-layer netowrk
        last_layer = len(shape) - 1
        for i, width in enumerate(shape):
            if i == 0: # the first layer need to specify input_shape
                model.add(LSTM(width, input_shape=(history_length, interval_width), consume_less="gpu", return_sequences=True))
            elif i == last_layer: #the last layer does not return sequences
                model.add(LSTM(width, consume_less="gpu"))
            else: # middle layers need return sequences, but not input shape
                model.add(LSTM(width, consume_less="gpu", return_sequences=True))
           
            if dropout > 0: # finally, if we are applying dropout, add it
                model.add(Dropout(dropout))
        
    #after all layers, add a dense layer and then the activation layer
    model.add(Dense(interval_width, activation=activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def get_GRU(shape, optimizer, loss, interval_width, history_length, activation, dropout=0):
    model = Sequential()
    if len(shape) == 1: # for a single layer network
            model.add(GRU(shape[0], input_shape=(history_length, interval_width), consume_less="gpu"))
            if dropout > 0:
                model.add(Dropout(dropout))
    else: # for a multi-layer netowrk
        last_layer = len(shape) - 1
        for i, width in enumerate(shape):
            if i == 0: # the first layer need to specify input_shape
                model.add(GRU(width, input_shape=(history_length, interval_width), consume_less="gpu", return_sequences=True))
            elif i == last_layer: #the last layer does not return sequences
                model.add(GRU(width, consume_less="gpu"))
            else: # middle layers need return sequences, but not input shape
                model.add(GRU(width, consume_less="gpu", return_sequences=True))
           
            if dropout > 0: # finally, if we are applying dropout, add it
                model.add(Dropout(dropout))
        
    #after all layers, add a dense layer and then the activation layer
    model.add(Dense(interval_width, activation=activation))

    model.compile(optimizer=optimizer, loss=loss)

    return model

def train_network(model, inputs, targets, epochs=1, callbacks=[], batch_size=512):
    model.fit(inputs, targets, nb_epoch=epochs, callbacks=callbacks, batch_size=batch_size)
    if model.stop_training is True:
        return model, False
    return model, True

def test_melody_network(model, seed_sequence, sequence_length, interval_width, history_length):
    generated_sequence = seed_sequence
    for i in range(sequence_length):
        # make a window into the generated sequence that is of the proper length
        test_sequence = np.zeros((1, history_length, interval_width), dtype=np.bool)
        for j in range(history_length):
            for k in range(interval_width):
                test_sequence[0, j, k] = generated_sequence[0, j + i, k]

        prediction = model.predict(test_sequence)[0]

        # pick the best note
        predicted_matrix = get_top_n(prediction, interval_width, n=1)
        generated_sequence = np.append(generated_sequence, predicted_matrix, 1)

    return generated_sequence

def test_accompaniment_network(model, melody, notes_to_select, history_length, threshold=.5):
    generated_sequence = []

    for i in range(len(history_length - 1)):
        generated_sequence.append(melody[i]) # start the accompaniment generation at the end of the seed to the melody generation

    for i in range(len(test_melody_network) - history_length): # the first 'history length' notes are the seed to the melody, so leave them alone.
        # make a window into the generated sequence that is of the proper length
        test_sequence = np.zeros((1, history_length, interval_width), dtype=np.bool)
        for j in range(history_length - 1): # history - 1 to allow adding the seed melody on.
            for k in range(interval_width):
                test_sequence[0, j, k] = generated_sequence[0, j + i, k]

        melody_note = -1
        for k in range(interval_width): # find the melody seed
            if melody[0, history_length + i, k] == 1:
                melody_note = k
                break
        if melody_note != -1: # if it is -1, there was a rest
            test_sequence[0, history_length - 1, melody_note] = 1

        prediction = model.predict(test_sequence)[0]
        predicted_matrix = get_above_percent(prediction, interval_width, percent=threshold)
        
        if melody_note != -1: # if there was a predicted melody note, make sure the accompaniment network did not lose it
            predicted_matrix[0, history_length - 1, melody_note] = 1

        generated_sequence = np.append(generated_sequence, predicted_matrix, 1)

    return generated_sequence

'''
def test_network(model, melody, notes_to_select, history_length, threshold=.5):
    for melody in melodies:
        generated_sequence = melody
        for i in range(sequence_length):
            # make a window into the generated sequence that is of the proper length
            test_sequence = np.zeros((1, history_length, interval_width), dtype=np.bool)
            for j in range(history_length):
                for k in range(interval_width):
                    test_sequence[0, j, k] = generated_sequence[0, j + i, k]

            prediction = model.predict(test_sequence)[0]
            predicted_matrix = get_above_percent(prediction, interval_width, percent=threshold)
            generated_sequence = np.append(generated_sequence, predicted_matrix, 1)

        generated_sequences.append(generated_sequence)

    return generated_sequences
    '''
def get_network_from_file(filename):
    model = None
    try:
        model = load_model(filename + ".h5")
    except: #in case the extension was specified
        model = load_model(filename)
    return model

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