notes = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B"]

def get_counts(state_matrix):
    totals = [0 for note in state_matrix[0]]

    for interval in state_matrix:
        for i, note in enumerate(interval):
            if note[1] == 1:
                totals[i] += 1


    totals = totals[4:] + totals[:4]
    mod_totals = [0 for i in range(12)]
    for i, val in enumerate(totals):
        mod_totals[i % 12] += val
     
    return mod_totals

def get_root(counts):
    root = 0
    max_score = 0
    for i, val in enumerate(counts):
        score = val + counts[(i + 7) % 12]

        if score > max_score:
            max_score = score
            root = i

    return root

def get_tone(root, counts):
    if counts[(root + 4) % 12] >= counts[(root + 3) % 12]:
        return "major"
    return "minor" 



def predict_key(state_matrix):
    counts = get_counts(state_matrix)
    print counts

    root = get_root(counts)

    tone = get_tone(root, counts)

    print(notes[root] + " " + tone)