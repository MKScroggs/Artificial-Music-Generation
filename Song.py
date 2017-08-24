notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def get_start_of_notes(matrix):
    """
    Finds the position of the first note hit (in time, not pitch)
    :param matrix: the matrix
    :return: the position of the first note
    """
    for i, state in enumerate(matrix):
        for note in state:
            if note[0] == 1:
                return i


def get_end_of_notes(matrix):
    """
    Finds the position of the last note hit (in time, not pitch)
    :param matrix: the matrix
    :return: the position of the last note
    """
    for i, state in reversed(list(enumerate(matrix))):
        for note in state:
            if note[0] == 1:
                return i + 1


def get_notes_from_interval(interval, include_pressed):
    """
    Gets the notes in an interval and returns a list of them, including pressed
    if include_pressed is True
    """
    return_interval = []
    for note in interval:
        return_interval.append(note[0])
        if include_pressed:
            return_interval.append(note[1])
    return return_interval


def get_beat_from_interval(interval, smallest_note, base_note, bpm):
    """
    Makes the tempo data from the interval count and the smallest_note
    """
    return_list = []

    intervals_per_measure = smallest_note / base_note * bpm
    for i in range(intervals_per_measure):
        if interval % smallest_note == i:
            return_list.append(1)
        else:
            return_list.append(0)
    return return_list


def fix_trailing_rests(song):
    """
    Ensures that a song has 0 blank states to start and to end. If not
    it makes it so.
    :param song: song to fix (they are almost never right to start...)
    :return: the fixed song
    """
    matrix = song.StateMatrix

    # create the blank state for
    blank_state = [[0, 0] for note in matrix[0]]

    # find the ends
    start_of_notes = get_start_of_notes(matrix)
    end_of_notes = get_end_of_notes(matrix)
    new_matrix = []

    for state in matrix[start_of_notes:end_of_notes]:
        new_matrix .append(state)

    song.StateMatrix = new_matrix
    return song


class Song(object):
    """
    Class that holds a song with its metadata
    """
    def __init__(self):
        self.TrackName = None
        self.StateMatrix = None
        self.MelodyStateMatrix = None
        self.SmallestNote = None
        self.Triplets = False
        self.BaseNote = None
        self.BeatsPerMeasure = None
        self.Tempo = None

    def get_training_matrix(self, mode="Melody", include_pressed=True,
                            include_beat=True, include_key=True,
                            include_tone=True):
        '''
        Creates a matrix for training with the song as well as additional
        meta-data. The format is a 2d list in the following order:
        [[note-down, note-pressed]...], [beat,...], [key...], [tone...]
        where note down is if the note is being pressed during the interval,
        note pressed is if the note is freshly pressed that interval,
        beat is what beat we are on represented by a list of 0's with a 1 on
        the current beat,
        key is a list of the keys, and tone is either major or minor
        :param mode: either melody or full. If melody, only the melody will be
        included.
        '''

        # prepare the matricies to be filled
        training_matrix = []
        matrix = []

        # if on melody mode, use only the melody, else use full song
        if mode == "Melody":
            if self.melody_matrix_set is False:
                self.set_melodymatrix_from_StateMatrix
            matrix = self.MelodyStateMatrix

        elif mode == "Full":
            matrix = self.StateMatrix
        else:
            # not a valid choice
            raise Exception("get_training_matrix not 'Melody' of 'Full'")

        # for each interval in the matrix
        song_len = len(matrix)
        print(song_len)
        for interval in range(song_len):
            new_interval = get_notes_from_interval(matrix[interval],
                                                   include_pressed)
 
            if include_beat:
                new_interval = new_interval + \
                               get_beat_from_interval(interval,
                                                      self.SmallestNote,
                                                      self.BaseNote,
                                                      self.BeatsPerMeasure)
                print(new_interval)

        return training_matrix

    def get_simple_matrix(self):
        simple_matrix = []
        for line in self.StateMatrix:
            simple_matrix.append([x[0] for x in line])
        return simple_matrix

    def get_simple_melody_matrix(self):
        if self.melody_matrix_set is False:
            self.set_melodymatrix_from_StateMatrix()
        simple_matrix = []
        for line in self.MelodyStateMatrix:
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

    def get_full_melody_matrix(self):
        if self.melody_matrix_set is False:
            self.set_melodymatrix_from_StateMatrix()
        full_matrix = []
        for line in self.MelodyStateMatrix:
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
                    new_line.append([0, 0])
                else:
                    if last_line[i] == 0:
                        new_line.append([1, 1])
                    else:
                        new_line.append([1, 0])
            last_line = new_line
            state_matrix.append(new_line)

        self.StateMatrix = state_matrix
        self.set_melodymatrix_from_StateMatrix()

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
        self.set_melodymatrix_from_StateMatrix()

    def set_melodymatrix_from_StateMatrix(self):
        melody_matrix = []

        blank_interval = [[0, 0] for note in self.StateMatrix[0]]

        last = -1
        # for each interval
        for i, interval in enumerate(self.StateMatrix):
            # add a blank interval
            melody_matrix.append([[0, 0] for note in self.StateMatrix[0]])

            # TODO: this should probably be made to run in reverse
            # find the highest note
            highest = -1
            for n, note in enumerate(interval):
                if note[0]:
                    # if the note is higher or equal than the prior highest
                    # note, record it
                    if n >= last:
                        highest = n

                    # if the note is lower, but also just pressed that
                    # interval, record it. If higher notes are pressed,
                    # it will get overwritten
                    elif note[1]:
                        highest = n

            # if highest == -1, then this is a rest, so dont record anythin
            if highest != -1:
                # now record the change
                melody_matrix[i][highest][0] = 1
                # record if it is a key press too
                if highest != last:
                    melody_matrix[i][highest][1] = 1
                    # if higest != last record a press, since the main song
                    # may not have it as such.
                else:
                    melody_matrix[i][highest][1] = interval[highest][1]
                    # if highest == last, record the songs data as it may be
                    # repeating notes.

            # finally record highest as the last note.
            last = highest

        # store the result
        self.MelodyStateMatrix = melody_matrix
        self.melody_matrix_set = True

    def transpose(self, transposition='auto', verbose=False):
        """
        Transposes a statematrix
        :param old_state_matrix: the matrix to transpose
        :param transposition: how to transpose it. -X shifts down X positions,
         +X shifts up X positions
        :return:
        """
        if transposition == 'auto':
            transposition = self.predict_key(verbose)

        if transposition is not 0:
            transposed_matrix = []
            for state in self.StateMatrix:
                transposed_matrix.append(state[transposition:] +
                                         state[:transposition])
            self.StateMatrix = transposed_matrix
            self.set_melodymatrix_from_StateMatrix()
        if verbose:
            print("Final state after transposition: ")
            self.predict_key(verbose)

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
            print(mod_counts)
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
            if note[1] == 1:
                # we are counting the presses, not note holds so note[1]
                # note note[0]
                counts[i] += 1

    counts = counts[1:] + counts[:1]
    # cycle the counts to center with the keys in the list of notes
    return counts


def get_mod_counts(counts):
    """
    takes a count of all notes per note in a song and converts to all
    occurences of a letter key (C_count = C1_count + C2_count + C3_count
    + C4_count ...)
    """
    mod_counts = [0 for i in range(12)]
    for i, val in enumerate(counts):
        mod_counts[i % 12] += val

    return mod_counts


def get_root(counts):
    """
    Predicts the root of a song based on the assumption that the sum of the
    counts of the root and the fifth will be higher than the sum of the same
    interval width starting on any other note.
    This assumption was tested and held true for all 32 of beethovens sonatas
    in testing.
    """
    root = 0
    max_score = 0
    for i, val in enumerate(counts):
        score = val + counts[(i + 7) % 12]
        # add the root and the 5th (7 semitones)

        if score > max_score:
            max_score = score
            root = i

    return root


def get_tone(root, counts):
    """
    Predicts the tone (major/minor) based on the counts of the major and minor
    third from the root. I assume that in a major song, the major third will
    be more common.
    This assumtion held true for 29 of the 32 Beethoven sonatas in testing.
    The three errors may have been due to resolution (midi) issues.
    """
    if counts[(root + 4) % 12] >= counts[(root + 3) % 12]:
        return "major"
    return "minor"\
