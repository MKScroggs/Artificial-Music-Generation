import Conversion
import keras.optimizers
import DataSets
import Networks
import Processing
import numpy as np
import os
import sys
from keras import metrics
from keras import callbacks
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


def build_new_network(shape=[512,512], learning_rate=.01, interval_width=88, history_length=16, 
                      loss="categorical_crossentropy", activation="softmax", dropout=.3):
    optimizer = keras.optimizers.RMSprop (lr=learning_rate)

    model = Networks.get_LSTM(shape, optimizer, loss, interval_width, history_length, activation, dropout=dropout, metrics=['accuracy'])
    return model

def load_existing_network(file_name):
    return Networks.get_network_from_file("Networks/" + file_name)


def train_network(model, data, epochs=1, callbacks=[], batch_size=512, interval_width=88, history_length=16, 
                  identifier="MISSING_IDENTIFIER", generation_length=1000, iterations=100):
    generated_dir = os.path.dirname(os.path.realpath(__file__)) + "/Txt/" + identifier
    network_dir = os.path.dirname(os.path.realpath(__file__)) + "/Networks/" + identifier
    for i in range(iterations):
        # if we haven't stopped learning
        print("\n... Iteration={0}".format(i))

        # train the model
        model = Networks.train_network(model, train_inputs=data.TrainingInput, train_targets=data.TrainingTarget, 
                                       val_inputs=data.TestInput, val_targets=data.TestTarget,
                                       epochs=epochs, callbacks=callbacks, batch_size=batch_size)
            
        # make sample songs and save them
        for seedcount, seed in enumerate(data.SeedInput):
            song = Networks.test_melody_network(model, seed, generation_length, interval_width, history_length)
            Processing.simple_nparray_to_txt(song, generated_dir + "_Iteration_{}_Seed_{}".format(i, seedcount), identifier + "_Iteration_{}".format(i))
            
        # save the network for reuse
        model.save(network_dir + "_Iteration_{}.h5".format(i))
        
        # if a callback has set stop training, then stop iterating
        if model.stop_training is True:
            break


def get_seed_data(data, history_length=16, data_sets=DataSets.seed):
    seed_sequences = Conversion.load_specified_state_matricies(data_sets)
    data.SeedInput = Processing.get_seed_data(seed_sequences, set_size=history_length)
    return data


def get_train_data(data, training_percent, history_length, data_sets=DataSets.simple_scales):
    training_songs = Conversion.load_specified_state_matricies(data_sets)
    return Processing.get_melody_training_data(training_songs, percent_to_train=training_percent, set_size=history_length + 1, data=data)


def full_run(learning_rate=.01, loss="categorical_crossentropy", activation="softmax", callbacks=[], interval_width=88, 
             history_length=16, batch_size=512, epochs=1, training_percent=.9, dropout=.3, iterations=100,
             seed_dataset=DataSets.seed, train_dataset=DataSets.beethoven_sonatas, generation_length=1000, shape=[512,512]):
    
    print("Starting up...")
    # do startup things like prepare txt files. Also get an identifier for the current test to use in file naming and network saving.
    identifier = startup()

    data = Processing.Data() # this holds the data we use. The functions that it is passed to fill its variables.

    print("Preparing Seed Data")
    data = get_seed_data(data=data, history_length=history_length, data_sets=seed_dataset)

    print("Preparing Training Data")
    data = get_train_data(data, training_percent, history_length, data_sets=train_dataset)
    
    print("Building network...")
    model = build_new_network(shape=shape, interval_width=interval_width, history_length=history_length, 
                      loss=loss, activation=activation, dropout=dropout, learning_rate=learning_rate)


    print("Starting training...")
    train_network(model, epochs=epochs, data=data, callbacks=callbacks, batch_size=batch_size, 
                  interval_width=88, history_length=history_length, identifier=identifier, generation_length=generation_length, iterations=iterations)
    
    close()


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except:
        print("Options are: 'midi' for midi-conversion, 'load [mode] [network]' to load a network (modes are -train and -test) and 'full' for a full training run")
        print("Running in 'full' mode")
        arg = "full"

    if arg == "midi":
        Conversion.convert_midi_folder(smallest_note=16)
        Conversion.convert_txt_folder()
    elif arg == "load":
        mode = None
        target = None
        try:
            mode = sys.argv[2]
            if mode != "-train" and mode != "-test":
                raise Exception()
        except:
            print("Invalid or empty field for 'mode'.")
        try:
            target = sys.argv[3]
        except:
            print("Empty field for 'network'.")
    else: # do a full run
        def learning_schedule(lr):
            return lr * .93

        learning_rate_callback = Networks.LearningRateCallback(learning_schedule)
        early_stoping_callback = callbacks.EarlyStopping(patience=1)
        full_run(iterations=200, callbacks=[learning_rate_callback, early_stoping_callback], learning_rate=.001, train_dataset=DataSets.full_data, history_length=16*4)
