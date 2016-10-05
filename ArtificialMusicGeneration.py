import Conversion
import keras.optimizers
import DataSets
import Networks
import Processing
import numpy as np
import os
from time import time
np.random.seed(6)


def startup():
    # convert the midi folder to ensure songs are present in txt form
    Conversion.convert_midi_folder(smallest_note=16)
    Conversion.convert_txt_folder()
   
    # return the time to use as an identifier in creating files
    return str(int(time()))
    

def close():
    
    Conversion.convert_txt_folder()


def main():
    interval_width = 88
    history_length = 12

    # do startup things like prepare txt files. Also get an identifier for the current test to use in file naming and network saving.
    identifier = startup()

    generated_dir = os.path.dirname(os.path.realpath(__file__)) + "/Txt/" + identifier

    # load training songs
    training_songs = Conversion.load_specified_state_matricies(DataSets.beethoven_sonatas)
    transposed_songs = []
    for song in training_songs:
        song.transpose()
    
    # convert training songs to network usable snippets 
    training_input, training_target, test_sequence = Processing.get_training_data(training_songs, set_size=history_length + 1)

    #Networks.main(training_input, training_target, test_sequence, identifier)
    model = Networks.get_SimpleRNN([50], "rmsprop", "mse", interval_width, history_length, "sigmoid")

    for i in range(50):
        print("... Iteration={0}".format(i))
        model, keep_going = Networks.train_network(model, training_input, training_target, epochs=5)
        songs = Networks.test_network(model, [test_sequence], 1, 50, interval_width, history_length)
        for song in songs:
            Processing.simple_nparray_to_txt(song, generated_dir + "_Iteration_{}".format(i), identifier + "_Iteration_{}".format(i))


    close()


if __name__ == "__main__":
    main()
