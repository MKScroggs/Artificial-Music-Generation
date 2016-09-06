# Modified from https://github.com/hexahedria/biaxial-rnn-music-composition/blob/master/midi_to_statematrix.py
import midi
import numpy
import os

lowerBound = 20
upperBound = 108
volume = 40


def load_specified_state_matricies(filenames):
    dir =  os.path.dirname(os.path.realpath(__file__))
    songs = []
    
    #load the songs
    for filename in filenames:
        songs.append(read_state_matrix_file(dir + '/Txt/' + filename))
    
    return songs

def convert_default_midi_folder():
    convert_midi_folder(os.path.dirname(os.path.realpath(__file__)))
    
def convert_default_txt_folder():
    convert_txt_folder(os.path.dirname(os.path.realpath(__file__)))


def convert_txt_folder(dir):

    print "Converting .Txt to .Mid..."

    midis = [os.path.splitext(file)[0] for file in os.listdir(dir + "/Midi") if file.endswith('.mid')]
    print "Files in midi directory are: {}".format(midis)

    txts = [os.path.splitext(file)[0] for file in os.listdir(dir + "/Txt") if file.endswith('.txt')]
    print "Files in txt directory are:  {}".format(txts)

    for item in [item for item in txts if item not in midis]:
        print "Converting {} to .mid".format(item)
        convert_to_midi(dir + "/Txt/" + item, dir + "/Midi/" + item)

    print "...Done convertring .Txt to .Mid\n"


def convert_midi_folder(dir, base_note=4, beats_per_measure=4, smallest_note=4, triplets=False):

    print "Converting .Mid to .Txt..."

    midis = [os.path.splitext(file)[0] for file in os.listdir(dir + "\\Midi") if file.endswith('.mid')]
    print "Files in midi directory are: {}".format(midis)

    txts = [os.path.splitext(file)[0] for file in os.listdir(dir + "\\Txt") if file.endswith('.txt')]
    print "Files in txt directory are:  {}".format(txts)

    for item in [item for item in midis if item not in txts]:
        print "Converting {} to .txt".format(item)
        convert_to_matrix(dir + "/Midi/" + item, dir + "/Txt/" + item,
                          base_note, beats_per_measure, smallest_note, triplets)

    print "...Done convertring .Mid to .Txt\n"


def convert_to_matrix(file_in, file_out, base_note=4, beats_per_measure=4, smallest_note=4, triplets=False, transpostion=0):
    # set vars for conversion
    time_signature = TimeSignature(base_note, beats_per_measure)
    desired_interval = DesiredInterval(smallest_note, triplets)

    # Read the Midi and return the song
    song = midi_to_note_state_matrix(file_in, file_out, time_signature, desired_interval)

    # remove extra rests from beggining and end
    song = fix_trailing_rests(song)

    song.transpose(transpostion)
    # Write the song to a file
    write_state_matrix_file(file_out, song)


def convert_to_midi(file_in, file_out):
    # read the song file and return the generated song
    song = read_state_matrix_file(file_in)

    # convert the song to a Midi file and write to the location file_out
    note_state_matrix_to_midi(song, file_out)


def write_state_matrix_file(filename, song):
    """
    :param filename: the name of the file to be written
    :param song: the song data (of type Song)
    :return: na
    """
    f = open(filename + '.txt', 'w')

    # write the header
    f.write(song.TrackName + '\n' + str(song.Tempo) + '\n' + str(song.TimeSignature.BaseNote) + '\n' +
               str(song.TimeSignature.BeatsPerMeasure) + '\n' + str(song.Interval) + '\n' +
               str(song.Resolution) + '\n')

    # get the length of the first line
    length = len(song.StateMatrix[0])

    # for each line...
    for state in song.StateMatrix:
        string = ''

        # add each note to the output
        for i in range(length):
            string += "{0};{1}".format(state[i][0], state[i][1])
            if i < length - 1:
                # if it is not the last note, add a comma
                string += ','
        string += '\n'

        # write the line
        f.write(string)

    f.close()


def read_state_matrix_file(filename):
    """
    Reads a statematrix file and generates a Song object for it
    :param filename: the file to read
    :return: a Song
    """
    # initialize the song
    song = None
    with open(filename + '.txt', 'r') as f:
        # read the header
        name = f.readline()
        tempo = int(f.readline())
        base_note = int(f.readline())
        beats_per_measure = int(f.readline())
        interval = int(f.readline())
        resolution = int(f.readline())

        # read the notes
        state_matrix = []
        for line in f:
            state = []
            parts = line.split(',')

            # add each note to the state for that instant
            for part in parts:
                subparts = part.split(';')
                state.append([int(subparts[0]), int(subparts[1])])
            state_matrix.append(state)

        # build the Song object
        song = Song(name, base_note, beats_per_measure, interval, tempo, numpy.asarray(state_matrix), resolution)

    return song







'''
############# Use the Code above in external functions. ####################
'''







class TimeSignature(object):
    """
    Simple class for the time signature to reduce parameter passing
    """
    def __init__(self, beats=4, base=4):
        self.BeatsPerMeasure = beats # in 2:4 time, this is the 2
        self.BaseNote = base # and this is the 4


class DesiredInterval(object):
    """
    Data used to determine how to read/write a file
    """
    def __init__(self, base=8, triplets = False):
        # what is the smallest note we will write (smaller notes will be read, but written as this
        self.SmallestNote = base
        # are we reading triplets?
        self.Triplets = triplets

    # gets the integer representation of the interval
    def getInt(self):
        if self.Triplets:
            return self.SmallestNote * 3
        return self.SmallestNote


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


def get_ticks_per_interval(resolution, time_signature, desired_interval):
    """
    NOTES ON MIDI Time Issues

    Midi uses ticks, which are non-constant in length between songs.
    to calculate ticks per beat we need to know Resolution (tempo provided too)

    Tempo is Beats per minute / quarter-notes per minute
    Resolution is pulses per quarter-note / ticks per beat

    Ticks per measure = Resolution * Quarter notes per measure
    """
    # get the ticks per measure
    ticks_per_measure = resolution * time_signature.BeatsPerMeasure

    # determine the number of intervals to train on per measure
    intervals_per_measure = time_signature.BeatsPerMeasure * desired_interval.SmallestNote / time_signature.BaseNote
    if desired_interval.Triplets is True:  # if triplets are to be read properly, multiply by 3
        intervals_per_measure *= 3

    # the ticks per interal is the ticks per measure / intervals per measure
    return int(ticks_per_measure / intervals_per_measure)


def midi_to_note_state_matrix(midifile, name, time_sig, desired_interval, tempo=90):
    """
    Reads a Midi File and generates a Song object to return
    :param midifile: The file to read
    :param name:  The name of the song
    :param time_sig: The time signature of the song (type TimeSignature)
    :param desired_interval: the desired minimum interval size (type DesiredInterval)
    :param tempo: The default tempo of the piece. For use if the Midi lacks a tempo. Defaults to 90.
    :return:
    """
    # read the midi file
    pattern = midi.read_midifile(midifile + ".mid")

    # get the number of ticks to read at a time
    ticks_per_interval = get_ticks_per_interval(pattern.resolution, time_sig, desired_interval)

    # how many ticks are there? for each track
    time_left = [track[0].tick for track in pattern]

    # initialize the positions of each track to 0
    posns = [0 for track in pattern]

    # this will be the song
    state_matrix = []

    # how many keys are we accepting
    span = upperBound - lowerBound

    # initialize the start time
    time = 0

    # create a black state (all 0's)
    state = [[0, 0] for x in range(span)]

    # add the state to the file. This will ensure all song
    state_matrix.append(state)

    while True:
        # for each track
        for i in range(len(time_left)):
            while time_left[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                # read the note events
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - lowerBound] = [0, 0]
                        else:
                            state[evt.pitch - lowerBound] = [1, 1]

                # Removed with songs including time signature
                # elif isinstance(evt, midi.TimeSignatureEvent):
                #    if evt.numerator not in (2, 4):
                #        # We don't want to worry about non-4 time signatures. Bail early!
                #        print "Found time signature event {}. Bailing!".format(evt)
                #        return state_matrix

                elif isinstance(evt, midi.SetTempoEvent):
                    # we want to get the tempo for recreation purposes. This does not handle multi tempo songs well.
                    # (just picks the last tempo)
                    tempo = int(evt.bpm)

                try:
                    time_left[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    time_left[i] = None

            if time_left[i] is not None:
                time_left[i] -= 1

        if all(t is None for t in time_left):
            break

        if time % ticks_per_interval == 0:
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            state_matrix.append(state)

        time += 1

    # build the song to return
    song = Song(name, time_sig.BaseNote, time_sig.BeatsPerMeasure, ticks_per_interval, tempo, state_matrix,
                pattern.resolution)
    return song


def note_state_matrix_to_midi(song, name):
    """
    Converts a song to Midi and writes to a file
    :param song: The song to convert
    :param name: The name of the Midi to create
    """
    statematrix = numpy.asarray(song.StateMatrix)
    pattern = midi.Pattern(resolution=song.Resolution)

    track = midi.Track()
    pattern.append(track)

    span = upperBound - lowerBound

    tickscale = song.Interval

    # Midi uses bytes to encode data. Tempo is in milliseconds, so it takes 3 bytes to encode the tempo.
    # I don't want to convert to the 3byte encoding, so i will use the setter provided by python-midi
    tempo_event = midi.SetTempoEvent(tick=0, data=[1, 1, 1])  # instantiate with nonsense data (tick is right though)
    tempo_event.set_bpm(song.Tempo)  # use setter to convert tempo (in bpm) to Midi encoding
    track.append(tempo_event)
    lastcmdtime = 0
    prevstate = [[0, 0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time - lastcmdtime) * tickscale, pitch=note + lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale, velocity=volume, pitch=note + lowerBound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)


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
                return i


def fix_trailing_rests(song):
    """
    Ensures that a song has 0 and only 0 blank states to start and 1 and only 1 blank state to end. If not, it makes it
    so.
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

    new_matrix.append(blank_state)

    song.StateMatrix = new_matrix
    return song

