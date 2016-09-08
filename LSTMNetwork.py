
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy
import pandas
import math

numpy.random.seed(6)

def get_training_data(songs, set_size=8, start=0, width=88):
    training_preceeding_intervals = []
    training_next_interval = []

    matricies = convert_data(songs, start, width)

    print("--- Building subsets ---")
    for i, matrix in enumerate(matricies):
        print("    Building subsets for song {}".format(i))
        length = len(matrix)

        for i in range(length - set_size + 1):
            preceeding_intervals = []
            for j in range(set_size - 1):
                preceeding_intervals.append(matrix[j])

            training_preceeding_intervals.append(preceeding_intervals)
            training_next_interval.append(matrix[i + set_size - 1])

    print('--- Vectorizing ---')
    # (how many datagroups, length of datagroups, width of intervals)
    preceeding = np.zeros((len(training_preceeding_intervals), set_size, width), dtype=np.bool)
    # (how many datagroups, width of intervals)
    next = np.zeros((len(training_preceeding_intervals), width), dtype=np.bool)
    for i, section in enumerate(training_preceeding_intervals):
        for t, interval in enumerate(section):
            pass
            #preceeding[i, t, char_indices[char]] = 1
       # next[i, char_indices[next_chars[i]]] = 1


def convert_data(dataset, start, width):
    print("--- Resizing matricies ---")
    training_data = []
    for song in dataset:
        matrix = song.get_simple_matrix()
        resized_matrix = []
        for line in matrix:
            resized_matrix.append(line[start:start + width])
        training_data.append(resized_matrix)
    print("--- Resizing complete ---")
    return training_data
    
def LSTM_main(songs):
    print("Converting training data ...")
    training_data = get_training_data(songs, set_size=8, start=40, width=13)



    pass

    