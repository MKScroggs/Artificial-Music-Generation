
from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
import pandas
import math
from time import time
import os
import Processing

np.random.seed(6)

def main(training_input, training_target, seed_sequence, identifier):

    # this is where we will save the resulting networks and music generated at each epoch
    network_dir = os.path.dirname(os.path.realpath(__file__)) + "/Networks/" + identifier
    generated_dir = os.path.dirname(os.path.realpath(__file__)) + "/Txt/" + identifier
    print("Building network model ... Time={}".format(time()))

    history_length = len(training_input[0])
    interval_width = len(training_input[0][0])
    model = Sequential()

    #add the LSTM layer 
    model.add(LSTM(50, input_shape=(history_length, interval_width)))
    model.add(Dense(interval_width))
    model.add(Activation('sigmoid'))

    optimizer = RMSprop(lr=0.001)

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    

    print("Fitting model ... Time={}".format(time()))

    for iteration in range(1, 251):
        print("... Iteration={0} ... time={1}".format(iteration, time()))
        model.fit(training_input, training_target, nb_epoch=1)

        if iteration % 25 == 0:
            model.save(network_dir + "_Iteration_{}".format(iteration))

            print("... Generating test sequence ...")
            generated_sequence = seed_sequence
            
            for i in range(100):
            
                test_sequence = np.zeros((1, history_length, interval_width), dtype=np.bool)
                for j in range(history_length):
                    for k in range(interval_width):
                        test_sequence[0, j, k] = generated_sequence[0, j + i, k]
      

                prediction = model.predict(test_sequence)[0]

                max_pred = 0
                index = 0
                for i, predicted_note in enumerate(prediction):
                    if predicted_note > max_pred:
                        max_pred = predicted_note
                        index = i

                predicted_matrix = np.zeros((1, 1, interval_width), dtype=np.bool)
                predicted_matrix[0, 0, index] = True
                generated_sequence = np.append(generated_sequence, predicted_matrix, 1)

            Processing.simple_nparray_to_txt(generated_sequence, generated_dir + "_Iteration_{}".format(iteration), identifier + "_Iteration_{}".format(iteration))

    return

    