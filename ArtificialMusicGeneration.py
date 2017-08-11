import Conversion
import keras.optimizers
import DataSets
import Networks
import Processing
import numpy as np
import os
import sys
from time import time
np.random.seed(6)


def build_new_network(shape=[512, 512], learning_rate=.01,
                      interval_width=88, history_length=16,
                      loss="categorical_crossentropy",
                      activation="softmax", dropout=.3):
    # optimizer = keras.optimizers.RMSprop (lr=learning_rate)

    optimizer = keras.optimizers.Adam()
    model = Networks.get_LSTM(shape, optimizer, loss, interval_width,
                              history_length, activation, dropout=dropout,
                              metrics=['accuracy'])

    return model


def load_existing_network(file_name):
    return Networks.get_network_from_file("Networks/" + file_name)


def train_network(model, data, epochs=1, callbacks=[], batch_size=512,
                  interval_width=88, history_length=16,
                  identifier="MISSING_IDENTIFIER", generation_length=1600,
                  iterations=100, mode="melody", feature_range=[1]):

    generated_dir = os.path.dirname(os.path.realpath(__file__)) +\
        "/Txt/" + identifier
    network_dir = os.path.dirname(os.path.realpath(__file__)) +\
        "/Networks/" + identifier

    # make a song to show randomized state (or start state if loaded \
    # from prior network)
    for seedcount, seed in enumerate(data.SeedInput):
        if mode == "melody":
            song = Networks.test_melody_network(model, seed, generation_length,
                                                interval_width, history_length)
            Processing.simple_nparray_to_txt(
                song, generated_dir +
                "_Iteration_0_Seed_{}".format(seedcount), identifier
                + "_Iteration_0")
        else:
            song = Networks.test_accompaniment_network(model, seed,
                                                       interval_width,
                                                       history_length)
            Processing.simple_nparray_to_txt(song, generated_dir +
                                             "_Iteration_0_Seed_{}".format(
                                                 seedcount), identifier
                                             + "_Iteration_0")

    # Start training
    for i in range(iterations):
        # if we haven't stopped learning
        print("\n... Iteration={0}".format(i + 1))

        # train the model
        model = Networks.train_network(model, train_inputs=data.TrainingInput,
                                       train_targets=data.TrainingTarget, 
                                       val_inputs=data.TestInput,
                                       val_targets=data.TestTarget,
                                       epochs=epochs, callbacks=callbacks,
                                       batch_size=batch_size)

        # make sample songs and save them
        for seedcount, seed in enumerate(data.SeedInput):
            for feature in feature_range:
                song = None
                if mode == "melody":
                    song = Networks.test_melody_network(model, seed,
                                                        generation_length,
                                                        interval_width,
                                                        history_length,
                                                        feature, 3)
                else:
                    song = Networks.test_accompaniment_network(model,
                                                               seed,
                                                               interval_width,
                                                               history_length,
                                                               feature, 3)
                Processing.simple_nparray_to_txt(
                    song, generated_dir +
                    "_Iteration_{}_Seed_{}_Feat_{}".format(
                        i + 1, seedcount.feature), identifier +
                    "_Iteration_{}".format(i + 1))

        # save the network for reuse
        model.save(network_dir + "_Iteration_{}.h5".format(i + 1))

        # if a callback has set stop training, then stop iterating
        if model.stop_training is True:
            break


def get_seed_data(data, history_length=16, data_sets=DataSets.seed):
    seed_sequences = Conversion.load_specified_state_matricies(data_sets)
    data.SeedInput = Processing.get_seed_data(seed_sequences,
                                              set_size=history_length)
    return data


def get_accompaniment_seed_data(data, data_sets=DataSets.seed):
    seed_sequences = Conversion.load_specified_state_matricies(data_sets)
    data.SeedInput = Processing.get_accompaniment_seed_data(seed_sequences)
    return data


def get_train_data(data, training_percent, history_length,
                   data_sets=DataSets.simple_scales, melody=True):
    training_songs = Conversion.load_specified_state_matricies(data_sets)
    if melody:
        return Processing.get_melody_training_data(
            training_songs,  percent_to_train=training_percent,
            set_size=history_length + 1, data=data)
    else:
        return Processing.get_accompaniment_training_data(
            training_songs, percent_to_train=training_percent,
            set_size=history_length + 1, data=data)


def full_melody_run(learning_rate=.01, loss="categorical_crossentropy",
                    activation="softmax", callbacks=[], interval_width=88,
                    history_length=16, batch_size=512, epochs=1,
                    training_percent=.9, dropout=.3, iterations=100,
                    seed_dataset=DataSets.seed,
                    train_dataset=DataSets.beethoven_sonatas,
                    generation_length=1000, shape=[512, 512]):

    print("Starting up...")
    # do startup things like prepare txt files. Also get an identifier for the
    # current test to use in file naming and network saving.
    identifier = startup()

    data = Processing.Data()
    # this holds the data we use. The functions that it is passed to fill
    # its variables.

    print("Preparing Seed Data")
    data = get_seed_data(data=data, history_length=history_length,
                         data_sets=seed_dataset)

    print("Preparing Training Data")
    data = get_train_data(data, training_percent, history_length,
                          data_sets=train_dataset)
    
    print("Building network...")
    model = build_new_network(shape=shape, interval_width=interval_width,
                              history_length=history_length, loss=loss,
                              activation=activation, dropout=dropout,
                              learning_rate=learning_rate)

    print("Starting training...")
    train_network(model, epochs=epochs, data=data, callbacks=callbacks,
                  batch_size=batch_size, interval_width=88,
                  history_length=history_length, identifier=identifier,
                  generation_length=generation_length, iterations=iterations)

    close()


def full_accompaniment_run(learning_rate=.01, loss="categorical_crossentropy",
                           activation="softmax", callbacks=[],
                           interval_width=88, history_length=16,
                           batch_size=1024, epochs=1,
                           training_percent=.9, dropout=.3, iterations=100,
                           seed_dataset=DataSets.seed,
                           train_dataset=DataSets.beethoven_sonatas,
                           generation_length=1000, shape=[512, 512]):

    print("Starting up...")
    # do startup things like prepare txt files. Also get an identifier for the
    # current test to use in file naming and network saving.
    identifier = startup()

    print(seed_dataset)
    data = Processing.Data()
    # this holds the data we use. The functions that it is
    # passed to fill its variables.

    print("Preparing Seed Data")
    data = get_accompaniment_seed_data(data=data, data_sets=seed_dataset)

    print("Preparing Training Data")
    data = get_train_data(data, training_percent, history_length,
                          data_sets=train_dataset, melody=False)
    
    print("Building network...")
    model = build_new_network(shape=shape, interval_width=interval_width,
                              history_length=history_length, loss=loss,
                              activation=activation, dropout=dropout,
                              learning_rate=learning_rate)

    print("Starting training...")
    train_network(model, epochs=epochs, data=data, callbacks=callbacks,
                  batch_size=batch_size, interval_width=88,
                  history_length=history_length,
                  identifier=identifier, generation_length=generation_length,
                  iterations=iterations, mode="accompaniment",
                  feature_range=[.25, .5, 1, 1.5])

    close()


def test_existing_melody_network(model, identifier, interval_width=88,
                                 history_length=96, seed_dataset=DataSets.seed,
                                 generation_length=10000, count=3):
    generated_dir = os.path.dirname(os.path.realpath(__file__)) + "/Txt/"
    + identifier.split('.')[0] + '_'
    data = Processing.Data()
    data = get_seed_data(data=data, history_length=history_length,
                         data_sets=seed_dataset)
    for seedcount, seed in enumerate(data.SeedInput):
        for temperature in [0.5, 0.75, 0.875, 1, 1.25, 1.5, 2]:
            print("Generating for seed:{} and temp:{}".format(seedcount,
                                                              temperature))
            song = Networks.test_melody_network(model, seed, generation_length,
                                                interval_width, history_length,
                                                temperature, count)
            Processing.simple_nparray_to_txt(
                song, generated_dir + "{}_Seed_{}_Temp_{}_Count_{}".format(
                    identifier, seedcount, temperature, count), identifier)


def test_existing_accompaniment_network(model, identifier, interval_width=88,
                                        history_length=96,
                                        seed_dataset=DataSets.seed,
                                        features=[.25, .5, .1],
                                        counts=[2, 3, 5]):
    generated_dir = os.path.dirname(os.path.realpath(__file__)) + "/Txt/" +\
                    identifier.split('.')[0] + '_'
    data = Processing.Data()
    data = get_accompaniment_seed_data(data=data, data_sets=seed_dataset)
    temperature = 1
    count = 1
    for seedcount, seed in enumerate(data.SeedInput):
        for temperature in features:
            for count in counts:
                print("Generating for seed:{} and temp:{}".format(seedcount,
                                                                  temperature))
                song = Networks.test_accompaniment_network(model, seed,
                                                           interval_width,
                                                           history_length,
                                                           temperature, count)
                Processing.simple_nparray_to_txt(
                    song, generated_dir + "Seed_{}_Temp_{}_Count_{}".format(
                        seedcount, temperature, count), identifier)


def test_full_accompaniment_run(
        learning_rate=.01, loss="categorical_crossentropy",
        activation="softmax", callbacks=[], interval_width=88,
        history_length=16, batch_size=1024, epochs=1, training_percent=1,
        dropout=.3, iterations=100, seed_dataset=DataSets.seed,
        train_dataset=DataSets.beethoven_sonatas, generation_length=1000,
        shape=[512, 512]):

    print("Starting up...")
    # do startup things like prepare txt files. Also get an identifier for the
    # current test to use in file naming and network saving.
    identifier = startup()

    print(seed_dataset)
    data = Processing.Data()  # this holds the data we use.
    # The functions that it is passed to fill its variables.

    print("Preparing Seed Data")
    seed_sequences = Conversion.load_specified_state_matricies(seed_dataset)
    data.SeedInput = Processing.get_test_seed_data(seed_sequences)

    print("Preparing Training Data")
    data = get_train_data(data, training_percent, history_length,
                          data_sets=train_dataset, melody=False)
    
    print("Building network...")
    model = build_new_network(shape=shape, interval_width=interval_width,
                              history_length=history_length, loss=loss,
                              activation=activation, dropout=dropout,
                              learning_rate=learning_rate)

    print("Starting training...")
    train_network(model, epochs=epochs, data=data, callbacks=callbacks,
                  batch_size=batch_size, interval_width=88,
                  history_length=history_length, identifier=identifier,
                  generation_length=generation_length, iterations=iterations,
                  mode="accompaniment", feature_range=[.5])

    close()


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
        mode = sys.argv[2]
    except:
        print("Options are: 'load [mode] [network]' to load a network "
              "(modes are -view and -test) and 'full [mode]' for a full "
              "training run (modes are -melody, -accomp, and -anneal)")
        raise Exception()
        
    if arg == "full":  # do a full run
        try:
            if mode != "-melody" and mode != "-accomp" and mode != "anneal":
                raise Exception()
        except:
            print("Invalid field for 'mode'.")
            raise Exception()
        def learning_schedule(lr):
            return lr * .9

        learning_rate_callback = Networks.LearningRateCallback(
            learning_schedule)

        if mode == "-melody":
            full_melody_run(shape=[512, 512], epochs=1, iterations=25,
                            callbacks=[learning_rate_callback],
                            learning_rate=.001, 
                            train_dataset=DataSets.sonatas,
                            seed_dataset=["minor_seed"],
                            history_length=16*6,
                            loss="categorical_crossentropy",
                            activation="softmax")
        elif mode == "-accomp":
            full_accompaniment_run(shape=[512, 512, 256], epochs=1,
                                   iterations=100, callbacks=[],
                                   learning_rate=.0001,
                                   train_dataset=DataSets.sonatas,
                                   seed_dataset=DataSets.small_melody_seed,
                                   history_length=int(16*1),
                                   loss="categorical_crossentropy",
                                   activation="softmax", batch_size=1026)
        else:
            raise NotImplementedError()
                                   
    elif arg == "load":
        target = None
        try:
            if mode != "-test" and mode != "-view":
                raise Exception()
        except:
            print("Invalid field for 'mode'.")
            raise Exception()
        try:
            target = "Networks/" + sys.argv[3]
        except:
            print("Empty field for 'network'.")
            raise Exception()
        if mode == "-view":
            Networks.view_network(Networks.get_network_from_file(target))
        elif mode == "-testa":
            model = Networks.get_network_from_file(target)
            print("Test_of_{}".format(sys.argv[3]))
            test_existing_accompaniment_network(
                model, "Test_of_{}".format(sys.argv[3]), history_length=16*2,
                seed_dataset=DataSets.small_melody_seed,
                features=[.5, 1, .75, 1.5])
        elif mode == "-testm":
            model = Networks.get_network_from_file(target)
            test_existing_melody_network(model,
                                         "Test_of_{}".format(sys.argv[3]),
                                         seed_dataset=["minor_seed",
                                                       "major_seed",
                                                       "Twinkle"])
