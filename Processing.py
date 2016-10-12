import Song
import Conversion
import numpy as np
from time import time
import os
import random

class Data(object):
    def __init__(self):
        self.TrainingInput  = []
        self.TrainingTarget = []
        self.TestInput      = []
        self.TestTarget     = []
        self.SeedInput      = []
        

def get_training_data(songs, percent_to_train, set_size=8, start=0, width=88):
    print("Building training data...")
    training_preceeding_intervals = []
    training_next_interval = []

    matricies = resize_dataset(songs, start, width)

    for i, matrix in enumerate(matricies):
        print("... Building subsets for song {}".format(i))
        length = len(matrix)

        for j in range(length - set_size + 1):
            preceeding_intervals = []
            for k in range(set_size - 1):
                preceeding_intervals.append(matrix[j + k])
            training_preceeding_intervals.append(preceeding_intervals)
            training_next_interval.append(matrix[j + set_size - 1])

    print("Vectorizing...")
    # (how many datagroups, length of datagroups, width of intervals)
    preceeding = np.zeros((len(training_preceeding_intervals), set_size - 1, width), dtype=np.bool)
    # (how many datagroups, width of intervals)
    next = np.zeros((len(training_preceeding_intervals), width), dtype=np.bool)

    for i, section in enumerate(training_preceeding_intervals):
        for t, interval in enumerate(section):
            for c, note in enumerate(interval):
                preceeding[i, t, c] = note
        for n, note in enumerate(training_next_interval[i]):
            next[i, n] = note

    return seperate_into_training_and_test(preceeding, next, percent_to_train)


def get_seed_data(songs, set_size=8, start=0, width=88):
    print("Building seed data...")
    sequences = []

    matricies = resize_dataset(songs, start, width)

    for i, matrix in enumerate(matricies):
        intervals = []
        for k in range(set_size - 1):
            intervals.append(matrix[k])
        sequences.append(intervals)

    # (how many datagroups, length of datagroups, width of intervals)
    seeds = np.zeros((len(training_preceeding_intervals), set_size, width), dtype=np.bool)

    for i, section in enumerate(sequences):
        for t, interval in enumerate(section):
            for c, note in enumerate(interval):
                seeds[i, t, c] = note

    return seeds


def seperate_into_training_and_test(preceeding, next, percent_to_train):
    random.seed(1)
    data = Data()
    for i in len(preceeding):
        if random.random() > percent_to_train:
            data.TestInput.append(preceeding[i])
            data.TestTarget.append(next[i])
        else:
            data.TrainingInput.append(preceeding[i])
            data.TrainingTarget.append(next[i])
    return data


def resize_dataset(dataset, start, width):
    print("Resizing matricies ... Time={}".format(time()))
    training_data = []
    for song in dataset:
        matrix = song.get_simple_matrix()
        resized_matrix = []
        for line in matrix:
            resized_matrix.append(line[start:start + width])
        training_data.append(resized_matrix)
    return training_data

def restore_song_size(song, start, width):
    beginning = [0 for i in range(start)]
    end = [0 for i in range(88 - start - width)]

    return [beginning + interval + end for interval in song]

    
    
def simple_nparray_to_txt(array, path, name):
    state_matrix = []

    song = Song.Song(name, 4, 4, 120, 120, None, 480)

    for i, interval in enumerate(array[0]):
        notes = []
        for j, note in enumerate(interval):
            notes.append(int(array[0, i, j]))
        state_matrix.append(notes)
    song.set_StateMatrix_from_simple_form(restore_song_size(state_matrix, 0, 88))

    Conversion.write_state_matrix_file(path, song)