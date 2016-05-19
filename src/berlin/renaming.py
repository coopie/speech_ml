import os

DATA_DIR = 'corpora/berlin/'

letter_to_emotion = {
    'W': 'angry',
    'E': 'disgust',
    'A': 'fearful',
    'F': 'happy',
    'T': 'sad',
    'N': 'neutral',
    'L': 'calm' # technically is boredom
}

def main():
    wavfiles = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    for f in wavfiles:
        os.rename(DATA_DIR + f, DATA_DIR + rename_file(f))


def rename_file(path):
    filename = path.split('/')[-1].split('.')[0]
    ident = 'berlin' + filename[0:2]
    activity = filename[2:5]
    emotion = letter_to_emotion[filename[5]]
    repitition = str(ord(filename[6]) - 96)

    return '_'.join([ident, emotion, activity, repitition]) + '.wav'




if __name__ == '__main__':
    main()
