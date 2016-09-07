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

    c.convert_specific_midi("Simple_Scale(Chromatic)",'test',smallest_note=8)
    # load training data
    #test_songs = MC.load_specified_state_matricies(DS.load_simple_scales())
       


    close()

if __name__ == "__main__":
    main()
