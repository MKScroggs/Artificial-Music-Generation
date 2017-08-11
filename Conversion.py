
# Modified from https://github.com/hexahedria/biaxial-rnn-music-composition/
# blob/master/midi_to_statematrix.py
import midi
import numpy
import os
import Song

lowerBound = 20
upperBound = 108
volume = 50

dir = os.path.dirname(os.path.realpath(__file__))



def load_song(filename, base_note=4, beats_per_measure=4,
              smallest_note=4, triplets=False):
    print("Loading {}".format(filename))
    # load the song
    file_location = dir + "/Midi/" + filename
    
    # set vars for conversion
    time_signature = Song.TimeSignature(base_note, beats_per_measure)
    desired_interval = DesiredInterval(smallest_note, triplets)

    # Read the Midi and return the song
    song = midi_to_song(file_location, filename, time_signature, 
                        desired_interval)
    
    return song


def save_song(song, filename):
	file_target = dir + "/Midi/" + filename
	song_to_midi(song, file_target)

class DesiredInterval(object):
    """
    Data used to determine how to read/write a file
    """
    def __init__(self, base=8, triplets=False):
        # what is the smallest note we will write
        self.SmallestNote = base
        # are we reading triplets?
        self.Triplets = triplets

    # gets the integer representation of the interval
    def getInt(self):
        if self.Triplets:
            return self.SmallestNote * 3
        return self.SmallestNote


def get_ticks_per_interval(resolution, time_signature, desired_interval):
    """
    NOTES ON MIDI Time Issues

    Midi uses ticks, which are non-constant in length between songs.
    to calculate ticks per beat we need to know Resolution (tempo provided too)

    Tempo is Beats per minute / quarter-notes per minute
    Resolution is pulses per quarter-note / ticks per beat

    Ticks per Measure = Resolution * Tempo
    """
    # get the ticks per measure
    ticks_per_measure = resolution * time_signature.BeatsPerMeasure

    # determine the number of intervals to train on per measure
    intervals_per_measure = time_signature.BeatsPerMeasure *\
                            desired_interval.SmallestNote /\
                            time_signature.BaseNote
    if desired_interval.Triplets is True:
        # if triplets are to be read properly, multiply by 3
        intervals_per_measure *= 3

    # the ticks per interal is the ticks per measure / intervals per measure
    return int(ticks_per_measure / intervals_per_measure)


def midi_to_song(midifile, name, time_sig, desired_interval,
                              tempo=100):
    """
    Reads a Midi File and generates a Song object to return
    :param midifile: The file to read
    :param name:  The name of the song
    :param time_sig: The time signature of the song (type TimeSignature)
    :param desired_interval: the desired minimum interval size
     (type DesiredInterval)
    :param tempo: The default tempo of the piece. For use if the Midi lacks a
     tempo. Defaults to 100.
    :return:
    """
    # read the midi file
    pattern = midi.read_midifile(midifile + ".mid")

    # get the number of ticks to read at a time
    ticks_per_interval = get_ticks_per_interval(pattern.resolution,
                                                time_sig, desired_interval)

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
                        # print "Note {} at time {} out of bounds
                        # (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or \
                           evt.velocity == 0:
                            state[evt.pitch - lowerBound] = [0, 0]
                        else:
                            state[evt.pitch - lowerBound] = [1, 1]

                elif isinstance(evt, midi.SetTempoEvent):
                    # we want to get the tempo for recreation purposes.
                    # This does not handle multi tempo songs well.
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
            # Crossed a note boundary. Create a new state,
            # defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            state_matrix.append(state)

        time += 1

    # build the song to return
    song = Song.Song(name, time_sig.BaseNote, time_sig.BeatsPerMeasure,
                     ticks_per_interval, tempo, state_matrix,
                     pattern.resolution)
    return song


def song_to_midi(song, name):
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
    """
    # Midi uses bytes to encode data. Tempo is in milliseconds, so it takes 3
    bytes to encode the tempo. I don't want to convert to the 3byte encoding,
    so i will use the setter provided by python-midi
    """
    # instantiate with nonsense data (tick is right though)
    tempo_event = midi.SetTempoEvent(tick=0, data=[1, 1, 1])
    # use setter to convert tempo (in bpm) to Midi encoding
    tempo_event.set_bpm(song.Tempo)

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
            track.append(midi.NoteOffEvent(tick=(time - lastcmdtime) *
                                           tickscale, pitch=note + lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) *
                                          tickscale, velocity=volume,
                                          pitch=note + lowerBound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
	
    print("{}.mid".format(name))
    midi.write_midifile("{}.mid".format(name), pattern)



