import sys
import os

DATA_DIR = "corpora/RAVDESS"

def main():
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.wav'):
            print(filename)
            os.rename(DATA_DIR + '/' + filename, DATA_DIR + '/' + get_new_name_RAVDESS(filename) + '.wav')


EMOTIONS = [
    None,
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

to_emotion = lambda x: EMOTIONS[x]

def get_new_name_RAVDESS(filename):
    flags = list(map(lambda x: int(x), (filename.split('.')[0]).split('-')))
    return  '_'.join((
        "RAVDESS" + str(flags[-1]),
        to_emotion(flags[2]) ,
        'kid' if flags[-3] == 1 else 'dog',
        str(flags[-2])
    ))

if __name__ == '__main__' and len(sys.argv) > 1:
    main()
