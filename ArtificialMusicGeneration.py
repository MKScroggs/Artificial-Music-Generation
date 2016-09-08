import Conversion as c
import DataSets as DS
import LSTMNetwork as LSTM
import numpy as np
import os

def startup():
    # convert the midi folder to ensure songs are present in txt form
    c.convert_default_midi_folder()
    

def close():
    c.convert_default_txt_folder()


def main():
    startup()

       
    # load training data
    test_songs = c.load_specified_state_matricies(DS.simple_scales)
    LSTM.LSTM_main(test_songs)


    #close()

if __name__ == "__main__":
    main()
