notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


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
        self.Tone = None

    def prepare_song(self, transpose=True):
        """
        Removes all rests preceeding and following the song. Also transposes to
        middle-c
        """
        self.fix_trailing_rests()
        if transpose:
            self.transpose()
        pass

    def get_training_matrix(self, greatest_bpm, mode="Melody",
                            include_pressed=True, include_beat=True,
                            include_tone=True, padding=16):
        '''
        Creates a matrix for training with the song as well as additional
        meta-data. The format is a 2d list in the following order:
        [[note-down, note-pressed]...], [beat,...], [tone...]
        where note down is if the note is being pressed during the interval,
        note pressed is if the note is freshly pressed that interval,
        beat is what beat we are on represented by a list of 0's with a 1 on
        the current beat,
        and tone is either major or minor
        :param mode: either melody or full. If melody, only the melody will be
        included.
        '''

        # prepare the matricies to be filled
        training_matrix = []
        matrix = []

        # if on melody mode, use only the melody, else use full song
        if mode == "Melody":
            if self.MelodyStateMatrix is None:
                self.extract_melody()
            matrix = self.MelodyStateMatrix
        elif mode == "Full":
            matrix = self.StateMatrix
        else:
            # not a valid choice
            raise Exception("get_training_matrix not 'Melody' of 'Full'")

        interval_width = len(matrix[0])

        # add the padding
        padding_matrix = [[[0, 0] for i in range(interval_width)]
                          for j in range(padding)]
        matrix = padding_matrix + matrix + padding_matrix

        # for each interval in the matrix
        song_len = len(matrix)

        for interval in range(song_len):
            new_interval = get_notes_from_interval(matrix[interval],
                                                   include_pressed)

            if include_beat:
                # subract padding to ensure that the actual song starts on the
                # 1 beat
                new_interval = new_interval + \
                               get_beat_from_interval(interval - padding,
                                                      self.SmallestNote,
                                                      self.BaseNote,
                                                      self.BeatsPerMeasure,
                                                      greatest_bpm)

            if include_tone:
                if self.Tone is None:
                    self.predict_tone()
                if self.Tone == "Major":
                    new_interval.append(1)
                else:
                    new_interval.append(0)
            training_matrix.append(new_interval)

        # add padding after interval to not have song position
        return training_matrix

    def fix_trailing_rests(self):
        """
        Ensures that a song has 0 blank states to start and to end. If not
        it makes it so.
        :param song: song to fix (they are almost never right to start...)
        """
        # find the ends
        start_of_notes = get_start_of_notes(self.StateMatrix)
        end_of_notes = get_end_of_notes(self.StateMatrix)
        new_matrix = []

        # slice the list at the start and end
        for state in self.StateMatrix[start_of_notes:end_of_notes]:
            new_matrix .append(state)

        self.StateMatrix = new_matrix

    def extract_melody(self):
        melody_matrix = []

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

    def transpose(self, transposition='auto'):
        """
        Transposes a statematrix
        :param old_state_matrix: the matrix to transpose
        :param transposition: how to transpose it. -X shifts down X positions,
         +X shifts up X positions
        :return:
        """
        if transposition == 'auto':
            transposition = self.predict_key()

        if transposition is not 0:
            transposed_matrix = []
            for state in self.StateMatrix:
                transposed_matrix.append(state[transposition:] +
                                         state[:transposition])
            self.StateMatrix = transposed_matrix

    def predict_key(self):
        """
        attempts to predict the key of the song for auto-transposition
        """
        counts = get_counts(self.StateMatrix)
        mod_counts = get_mod_counts(counts)

        root = get_root(mod_counts)
        transposition = root - 3  # takes into account the fact that key 20 is
        # not a C key
        return transposition

    def predict_tone(self):
        """
        attempts to predict the tone (major/minor) of the song
        """
        counts = get_counts(self.StateMatrix)
        mod_counts = get_mod_counts(counts)

        root = get_root(mod_counts)

        # here for fun, not practical use.
        self.Tone = get_tone(root, mod_counts)


def get_counts(state_matrix):
    """
    takes a state matrix and counts all the occurences of each note
    """
    counts = [0 for note in state_matrix[0]]

    for interval in state_matrix:
        for i, note in enumerate(interval):
            if note[1] == 1:
                # we are counting the presses, not note holds so note[1]
                # not note[0]
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
    #print(notes[root])
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
        return "Major"
    return "Minor"


def display_matrix(matrix):
    """
    Displays a matrix in an easier to read format (X and ' '). For debug use.
    """
    for line in matrix:
        display_line = []
        for item in line:
            if item == 1:
                display_line.append("X")
            else:
                display_line.append(" ")
        print(display_line)


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


def get_beat_from_interval(interval, smallest_note, base_note, bpm,
                           greatest_bpm):
    """
    Makes the beat data from the interval count and the smallest_note
    :param greatest_bpm: This is the largest bpm in all the songs being used,
    not just the current song. It is used to pad 0's to ensure all songs are of
    the same width.
    """
    return_list = []

    # used for modular math to ensure correct beats
    intervals_per_measure = smallest_note / base_note * bpm
    # used to ensure that each song is padded to the same length
    greatest_intervals_per_measure = smallest_note / base_note * greatest_bpm
    for i in range(greatest_intervals_per_measure):
        if interval % intervals_per_measure == i:
            return_list.append(1)
        else:
            return_list.append(0)
    return return_list
