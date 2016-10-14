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
    history_length = 16 * 6
    batch_size = 1024
    epochs = 1
    training_percent = .9
    save_network = True
    print("Starting up...")
    # do startup things like prepare txt files. Also get an identifier for the current test to use in file naming and network saving.
    identifier = startup()
    generated_dir = os.path.dirname(os.path.realpath(__file__)) + "/Txt/" + identifier
    network_dir = os.path.dirname(os.path.realpath(__file__)) + "/Networks/" + identifier

    print("Loading songs...")
    # load training songs
    seed_sequences = Conversion.load_specified_state_matricies(DataSets.seed)
    training_songs = Conversion.load_specified_state_matricies(DataSets.beethoven_sonatas)

    print("Making training data...")
    # convert training songs to network usable snippets 
    data = Processing.get_training_data(training_songs, percent_to_train=training_percent, set_size=history_length + 1)
    data.SeedInput = Processing.get_seed_data(seed_sequences, set_size=history_length)
    print("Building network...")
    optimizer = keras.optimizers.RMSprop (lr=.001)

    def learning_schedule(lr):
        return lr * .50

    learning_rate_callback = Networks.LearningRateCallback(learning_schedule)

    model = Networks.get_LSTM([512, 512], optimizer, "binary_crossentropy", interval_width, history_length, "sigmoid", dropout=.3, metrics=['accuracy'])

    print("Starting training...")
    keep_going_train = True
    keep_going_test = True
    for i in range(50):
        # if we haven't stopped learning
        print("\n... Iteration={0}".format(i))

        # train the model
        model, keep_going_train = Networks.train_network(model, data.TrainingInput, data.TrainingTarget, epochs=epochs, callbacks=[learning_rate_callback], batch_size=batch_size)
            
        # test the model
        score = model.evaluate(data.TestInput, data.TestTarget, batch_size=batch_size)
        keep_going_test = True
        print("The score is {}".format(score))

        # make a sample song
        songs = Networks.test_network(model, data.SeedInput, 6, 1000, interval_width, history_length, threshold=(.25))
            
        # convert generated songs to txt files
        for song in songs:
            Processing.simple_nparray_to_txt(song, generated_dir + "_Iteration_{}".format(i), identifier + "_Iteration_{}".format(i))

        # save the network for reuse
        if save_network:
            name = network_dir + "_Iteration_{}".format(i)
            model.save(name)
        
        if keep_going_train is False or keep_going_test is False:
            break
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
