import MidiConversion as MC
import os

simple_scale_names = [
    "Simple_Scale(Chromatic)",
    "Simple_Scale(Dorian)",
    "Simple_Scale(Harmonic_Minor)",
    "Simple_Scale(Hungarian_Minor)",
    "Simple_Scale(Ionian)",
    "Simple_Scale(Locrian)",
    "Simple_Scale(Dorian)",
    "Simple_Scale(Mixolydian)",
    "Simple_Scale(Phrygian)"
]


def main():
    # start by converting 
    MC.convert_default_midi_folder()
    test_songs = MC.load_specified_state_matricies(simple_scale_names)
    pass

if __name__ == "__main__":
    main()
