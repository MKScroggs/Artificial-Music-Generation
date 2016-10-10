
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
    def __init__(self, update, monitor='loss', patience=0, stop=3):
        super(LearningRateCallback, self).__init__()
        
        self.update = update
        self.monitor = monitor
        self.patience = patience

        self.wait = 0
        self.best = np.Inf
        self.stop = 3

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
            print("\nCurrent: {}, Best: {}".format(current, self.best))
            if self.wait >= self.patience:
                if self.wait >= self.stop:
                    self.model.stop_training = True
                    print("Stopping Early!")
                    return
                new_lr = self.update(backend.get_value(self.model.optimizer.lr))
                backend.set_value(self.model.optimizer.lr, new_lr)
                
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

def test_network(model, seed_sequences, notes_to_select, sequence_length, interval_width, history_length, threshold=.5):
    generated_sequences = []
    for seed_sequence in seed_sequences:
        generated_sequence = seed_sequence
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

def get_network_from_file(filename):
    model = None
    try:
        model = load_model(filename + ".h5")
    except: #in case the extension was specified
        model = load_model(filename)
    return model

def get_top_n(prediction, interval_width, n=1):
    indeces = np.argpartition(prediction, -notes_to_select)[-notes_to_select:]
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
    

# from keras
def fbeta_score(y_true, y_pred, beta=1):
    '''Compute F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification where input samples can be
    tagged with a set of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score
"""
def main(training_input, training_target, seed_sequence, identifier, notes_to_select=1, save_network=False):
    # this is where we will save the resulting networks and music generated at each epoch
    network_dir = os.path.dirname(os.path.realpath(__file__)) + "/Networks/" + identifier
    generated_dir = os.path.dirname(os.path.realpath(__file__)) + "/Txt/" + identifier
    print("Building network model ... Time={}".format(time()))

    history_length = len(training_input[0])
    interval_width = len(training_input[0][0])
    print history_length
    print interval_width
    model = Sequential()

    #add the LSTM layer 
    model.add(SimpleRNN(88, input_shape=(history_length, interval_width)))
    model.add(Dense(interval_width))
    model.add(Activation('softmax'))
    
    optimizer = RMSprop(lr=.01)
   # optimizer = SGD(lr=.01, momentum=.9, decay=.10, nesterov=True)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    def learning_schedule(lr):
        return lr * .5

    learning_rate_callback = LearningRateCallback(learning_schedule)
    

    print("Fitting model ... Time={}".format(time()))

    for iteration in range(1, 200):
        print("... Iteration={0} ... time={1}".format(iteration, time()))
        model.fit(training_input, training_target, nb_epoch=1, callbacks=[learning_rate_callback])
        if save_network:
            name = network_dir + "_Iteration_{}".format(iteration)
            model.save(name)

        print("... Generating test sequence ...")
        generated_sequence = seed_sequence
            
        for i in range(50):
            
            test_sequence = np.zeros((1, history_length, interval_width), dtype=np.bool)
            for j in range(history_length):
                for k in range(interval_width):
                    test_sequence[0, j, k] = generated_sequence[0, j + i, k]
      

            prediction = model.predict(test_sequence)[0]

            indeces = np.argpartition(prediction, -notes_to_select)[-notes_to_select:]
            predicted_matrix = np.zeros((1, 1, interval_width), dtype=np.bool)

            for index in indeces:
                predicted_matrix[0, 0, index] = True

            generated_sequence = np.append(generated_sequence, predicted_matrix, 1)

        Processing.simple_nparray_to_txt(generated_sequence, generated_dir + "_Iteration_{}".format(iteration), identifier + "_Iteration_{}".format(iteration))
            
    return

  """
