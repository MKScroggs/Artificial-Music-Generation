
from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import Callback
from keras import backend
from keras.utils import np_utils
import numpy as np
import pandas
import math
from time import time
import os
import Processing
import Conversion

np.random.seed(6)

class LearningRateCallback(Callback):
    def __init__(self, update, monitor='loss', patience=1, stop=3):
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
            #    if self.wait > self.stop:
             #       self.model.stop_training = True
            self.wait += 1

def main(training_input, training_target, seed_sequence, identifier, notes_to_select=3):

    # this is where we will save the resulting networks and music generated at each epoch
    network_dir = os.path.dirname(os.path.realpath(__file__)) + "/Networks/" + identifier
    generated_dir = os.path.dirname(os.path.realpath(__file__)) + "/Txt/" + identifier
    print("Building network model ... Time={}".format(time()))

    history_length = len(training_input[0])
    interval_width = len(training_input[0][0])
    model = Sequential()

    #add the LSTM layer 
    model.add(LSTM(88, input_shape=(history_length, interval_width), return_sequences=True, consume_less="cpu"))
    model.add(LSTM(88, return_sequences=True, consume_less="cpu"))
    model.add(LSTM(88, consume_less="cpu"))
    model.add(Dense(interval_width))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=.005)
   # optimizer = SGD(lr=.01, momentum=.9, decay=.10, nesterov=True)

    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    def learning_schedule(lr):
        return lr * .5

    learning_rate_callback = LearningRateCallback(learning_schedule)
    

    print("Fitting model ... Time={}".format(time()))

    for iteration in range(1, 1000):
        print("... Iteration={0} ... time={1}".format(iteration, time()))
        model.fit(training_input, training_target, nb_epoch=1, callbacks=[learning_rate_callback])
        name = network_dir + "_Iteration_{}".format(iteration)
        model.save(name)

        print("... Generating test sequence ...")
        generated_sequence = seed_sequence
            
        for i in range(480):
            
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
            
        if model.stop_training is True:
            return
    return

    