class Song(object):
    """
    Class that holds a song with its metadata
    """
    def __init__(self, name, base_note, beats_per_base_note, interval, tempo, state_matrix, resolution):
        self.TrackName = name
        self.TimeSignature = TimeSignature(base_note, beats_per_base_note)
        self.Tempo = tempo
        self.Interval = interval
        self.StateMatrix = state_matrix
        self.Resolution = resolution

    def transpose(self, transposition): 
        """
        Transposes a statematrix
        :param old_state_matrix: the matrix to transpose
        :param transposition: how to transpose it. -X shifts down X positions, +X shifts up X positions
        :return:
        """
        transposed_matrix = []
        for state in self.StateMatrix:
            transposed_matrix.append(state[-transposition:] + state[:-transposition])

        self.StateMatrix = transposed_matrix

    def get_simple_matrix(self):
        simple_matrix = []
        for line in self.StateMatrix:
            pass 
        pass

    def get_full_matrix():
        pass


class TimeSignature(object):
    """
    Simple class for the time signature to reduce parameter passing
    """
    def __init__(self, beats=4, base=4):
        self.BeatsPerMeasure = beats # in 2:4 time, this is the 2
        self.BaseNote = base # and this is the 4

