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
            simple_matrix.append([x[0] for x in line])
        return simple_matrix

    def get_full_matrix(self):
        full_matrix = []
        for line in self.StateMatrix:
            new_line = []
            for pair in line:
                new_line.append(pair[0])
                new_line.append(pair[1])
            full_matrix.append(new_line)
        return full_matrix

    def set_StateMatrix_from_simple_form(self, simple_matrix):
        state_matrix = []

        length = len(simple_matrix[0])
        last_line = [0 for i in range(length)]
        for line in simple_matrix:
            new_line = []
            for i in range(length):
                if line[i] == 0:
                    new_line.append([0,0])
                else:
                    if last_line[i] == 0:
                        new_line.append([1,1])
                    else:
                        new_line.append([1,0])
            state_matrix.append(new_line)
            
        self.StateMatrix = state_matrix 

    def set_StateMatrix_from_full_form(self, full_matrix):
        state_matrix = []

        # divide by two as every two notes is a pair of hold and press
        length = len(full_matrix[0])/2
        
        for line in full_matrix:
            new_line = []
            for i in range(length):
                index = i * 2
                new_line.append([line[index], line[index + 1]])
            state_matrix.append(new_line)

        self.StateMatrix = state_matrix


class TimeSignature(object):
    """
    Simple class for the time signature to reduce parameter passing
    """
    def __init__(self, beats=4, base=4):
        self.BeatsPerMeasure = beats # in 2:4 time, this is the 2
        self.BaseNote = base # and this is the 4

