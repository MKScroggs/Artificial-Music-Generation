
from keras.layers import LSTM, GRU, Dense, Activation, Dropout, SimpleRNN
from keras.models import Sequential, load_model
import keras.optimizers
from keras.callbacks import Callback
from keras import backend
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
                lr = backend.get_value(self.model.optimizer.lr)
                backend.set_value(self.model.optimizer.lr, self.update(lr))
                
                print("\nNew Learning rate: {}".format(lr)) 
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


def get_LSTM(shape, optimizer, loss, interval_width, history_length, activation, dropout=0):
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

    model.compile(optimizer=optimizer, loss=loss)

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

def test_network(model, seed_sequences, notes_to_select, sequence_length, interval_width, history_length):
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

            indeces = np.argpartition(prediction, -notes_to_select)[-notes_to_select:]
            predicted_matrix = np.zeros((1, 1, interval_width), dtype=np.bool)

            for index in indeces:
                predicted_matrix[0, 0, index] = True

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
