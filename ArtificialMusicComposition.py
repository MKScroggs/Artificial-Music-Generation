import Conversion as c
import DataSets as DS
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
    test_songs = c.load_specified_state_matricies(DS.simplest_scale)
    test_song = test_songs[0]
    test = test_songs[0].get_simple_matrix()

    print test_song.StateMatrix
    test_song.set_StateMatrix_from_full_form(test_song.get_full_matrix())
    print test_song.StateMatrix
       


    close()

if __name__ == "__main__":
    main()
