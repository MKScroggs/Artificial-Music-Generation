import Conversion
import DataSets
import LSTMNetwork
import Processing
import numpy as np
import os
from time import time

def startup():
    # convert the midi folder to ensure songs are present in txt form
    Conversion.convert_default_midi_folder()
   
    # return the time to use as an identifier in creating files
    return str(int(time()))
    

def close():
    
    Conversion.convert_default_txt_folder()


def main():
    # do startup things like prepare txt files. Also get an identifier for the current test to use in file naming and network saving.
    identifier = startup()

       
    # load training songs
    training_songs = Conversion.load_specified_state_matricies(DataSets.simple_scales)

    # convert training songs to network usable snippets 
    training_input, training_target, test_sequence = Processing.get_training_data(training_songs, set_size=8, start=40, width=13)

    # pass data to network for training and genertation
    LSTMNetwork.main(training_input, training_target, test_sequence, identifier)


    close()

if __name__ == "__main__":
    main()
