import MidiConversion as MC
import DataSets as DS

def main():
    # start by converting the midi folder to ensure songs are present in txt form
    MC.convert_default_midi_folder()
    
    # load training data
    test_songs = MC.load_specified_state_matricies(DS.simple_scale_names)

if __name__ == "__main__":
    main()
