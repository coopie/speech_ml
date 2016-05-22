import os

# corpora\toronto\OAF_back_angry.wav
# corpora\RAVDESS\RAVDESS1_angry_dog_1.wav


    # "neutral",
    # "calm",
    # "happy",
    # "sad",
    # "angry",
    # "fearful",
    # "disgust",
    # "surprised"

DIR = 'corpora/toronto/'


def main():
    for filename in os.listdir(DIR):
        if filename.endswith('.wav'):
            os.rename(DIR + filename, DIR + get_new_name(filename))



def to_emotion(toronto_emotion):
    if toronto_emotion == 'ps':
        toronto_emotion = 'surprised'

    return toronto_emotion


def get_new_name(filename):

    ident, activity, emotion = filename.split('.')[0].split('_')
    emotion = to_emotion(emotion)

    return '_'.join([ident, emotion, activity, '1']) + '.wav'




    return new_name


if __name__ == '__main__':
    main()
