import Conversion
import keras.optimizers
import DataSets
import Networks
import Processing
import numpy as np
import os
import sys
from keras import metrics
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
    history_length = 32
    save_network = True
    print("Starting up...")
    # do startup things like prepare txt files. Also get an identifier for the current test to use in file naming and network saving.
    identifier = startup()
    generated_dir = os.path.dirname(os.path.realpath(__file__)) + "/Txt/" + identifier
    network_dir = os.path.dirname(os.path.realpath(__file__)) + "/Networks/" + identifier
    print("Loading songs...")
    
    # load training songs
    training_songs = Conversion.load_specified_state_matricies(DataSets.beethoven_sonatas)
    for song in training_songs:
        print("\n\n" + song.TrackName)
        song.transpose(verbose=True)

    return
    print("Making training data...")
    # convert training songs to network usable snippets 
    training_input, training_target, test_sequence = Processing.get_training_data(training_songs, set_size=history_length + 1)
    
    print("Building network")
    optimizer = keras.optimizers.RMSprop (lr=.001)

    def learning_schedule(lr):
        return lr * .50

    learning_rate_callback = Networks.LearningRateCallback(learning_schedule)

    model = Networks.get_LSTM([1024,1024], optimizer, "binary_crossentropy", interval_width, history_length, "sigmoid", dropout=.1)
    print("Starting training...")
    keep_going = True
    for i in range(100):
        if keep_going:
            print("... Iteration={0}".format(i))
            model, keep_going = Networks.train_network(model, training_input, training_target, epochs=5, callbacks=[learning_rate_callback], batch_size=1024)
            songs = Networks.test_network(model, [test_sequence], 6, 1000, interval_width, history_length, threshold=(.25))
            for song in songs:
                Processing.simple_nparray_to_txt(song, generated_dir + "_Iteration_{}".format(i), identifier + "_Iteration_{}".format(i))
        if save_network:
            name = network_dir + "_Iteration_{}".format(i)
            model.save(name)


    close()


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except:
        arg = "-full"

    if arg == "-h":
        print("Options are: -h for help, -midi for midi-conversion, and blank for program")
    elif arg == "-midi":
        Conversion.convert_midi_folder(smallest_note=16)
        Conversion.convert_txt_folder()
    else:
        main()
