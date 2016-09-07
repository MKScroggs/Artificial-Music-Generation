"""
Wrapper for MidiConversion for simplicity
"""

import MidiConversion as MC
import os



def load_specified_state_matricies(filenames):
    dir =  os.path.dirname(os.path.realpath(__file__))
    songs = []
    
    #load the songs
    for filename in filenames:
        songs.append(MC.read_state_matrix_file(dir + '/Txt/' + filename))
    
    return songs

def convert_default_midi_folder(base_note=4, beats_per_measure=4, smallest_note=4, triplets=False):
    MC.convert_midi_folder(os.path.dirname(os.path.realpath(__file__)), base_note, beats_per_measure, smallest_note, triplets)
    
def convert_default_txt_folder():
    MC.convert_txt_folder(os.path.dirname(os.path.realpath(__file__)))

def convert_specific_midi(file_in, song_name, base_note=4, beats_per_measure=4, smallest_note=4, triplets=False, transposition=0):
    dir = os.path.dirname(os.path.realpath(__file__))
    MC.convert_to_matrix(dir + '/Midi/' + file_in, dir + '/Txt/' + song_name, base_note, beats_per_measure, smallest_note, triplets, transposition)
    pass