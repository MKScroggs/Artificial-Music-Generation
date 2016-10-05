notes = [
    "A",
    "A#",
    "B",
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#"
]

class TimeSignature(object):
    """
    Simple class for the time signature to reduce parameter passing
    """
    def __init__(self, beats=4, base=4):
        self.BeatsPerMeasure = beats # in 2:4 time, this is the 2
        self.BaseNote = base # and this is the 4


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
            last_line = new_line
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
        
    def transpose(self, transposition='auto'): 
        """
        Transposes a statematrix
        :param old_state_matrix: the matrix to transpose
        :param transposition: how to transpose it. -X shifts down X positions, +X shifts up X positions
        :return:
        """
        if transposition == 'auto':
            transposition = self.predict_key()

        if transposition is not 0:
            transposed_matrix = []
            for state in self.StateMatrix:
                transposed_matrix.append(state[transposition:] + state[:transposition])
            self.StateMatrix = transposed_matrix
    
    def predict_key(self, verbose=False):
        """
        attempts to predict the key of the song for auto-transposition
        """
        counts = get_counts(self.StateMatrix)
        mod_counts = get_mod_counts(counts)

        root = get_root(mod_counts)

        # here for fun, not practical use. 
        tone = get_tone(root, mod_counts)


        if verbose:
            print(notes[root] + " " + tone)
        transposition = root - 3
        return transposition


        
        
def get_counts(state_matrix):
    """
    takes a state matrix and counts all the occurences of each note
    """
    counts = [0 for note in state_matrix[0]]

    for interval in state_matrix:
        for i, note in enumerate(interval):
            if note[1] == 1: # we are counting the presses, not note holds so note[1] note note[0]
                counts[i] += 1
    
    counts = counts[1:] + counts[:1] # cycle the counts to center with the keys in the list of notes
    return counts

def get_mod_counts(counts):
    """
    takes a count of all notes per note in a song and converts to all occurences of a letter key (C_count = C1_count + C2_count + C3_count + C4_count ...)
    """
    mod_counts = [0 for i in range(12)]
    for i, val in enumerate(counts):
        mod_counts[i % 12] += val
     
    return mod_counts

def get_root(counts):
    """
    Predicts the root of a song based on the assumption that the sum of the counts of the root and the fifth will be higher than the sum of the same interval width
    starting on any other note.
    This assumption was tested and held true for all 32 of beethovens sonatas in testing.
    """
    root = 0
    max_score = 0
    for i, val in enumerate(counts):
        score = val + counts[(i + 7) % 12] # add the root and the 5th (7 semitones)

        if score > max_score:
            max_score = score
            root = i

    return root

def get_tone(root, counts):
    """
    Predicts the tone (major/minor) based on the counts of the major and minor third from the root. I assume that in a major song, the major third will be more common.
    This assumtion held true for 29 of the 32 Beethoven sonatas in testing. The three errors may have been due to resolution (midi) issues.
    """
    if counts[(root + 4) % 12] >= counts[(root + 3) % 12]:
        return "major"
    return "minor" 
