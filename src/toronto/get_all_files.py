import code
import urllib
from keras.utils.generic_utils import Progbar
from requests import get
from subprocess import call

BASE_URL = 'https://tspace.library.utoronto.ca/bitstream/1807/'

SUBJECTS = ['OAF', 'YAF']

WORDS = open('src/TORONTO/spoken_words.txt').read().split('\n')
WORDS.reverse()
WORDS = WORDS[1:]

print(WORDS)

EMOTIONS = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'ps' # pleasant surprise
]

DATA_DIR = 'corpora/TORONTO/'

NUM_LINKS = 2 * 200 * 7

def main():

    # combinations = [[x] for x in SUBJECTS]
    # combinations = add_permutaions(combinations, WORDS)
    # combinations = add_permutaions(combinations, EMOTIONS)

    # pb = Progbar(2 * 200 * 7)

    # for combination in combinations:
    #     name = '_'.join(combination) + '.wav'
    #     url = BASE_URL + name
    #     get_file(url, DATA_DIR + name)
    #     pb.add(1)
    pb = 2

    get_files_for_experiment(24499 ,'OAF', EMOTIONS[0], pb)
    get_files_for_experiment(24494 ,'OAF', EMOTIONS[1], pb)
    get_files_for_experiment(24492 ,'OAF', EMOTIONS[2], pb)
    get_files_for_experiment(24501 ,'OAF', EMOTIONS[3], pb)
    get_files_for_experiment(24488 ,'OAF', EMOTIONS[4], pb)
    get_files_for_experiment(24497 ,'OAF', EMOTIONS[5], pb)
    get_files_for_experiment(24491 ,'OAF', EMOTIONS[6], pb)

    get_files_for_experiment(24490 ,'YAF', EMOTIONS[0], pb)
    get_files_for_experiment(24498 ,'YAF', EMOTIONS[1], pb)
    get_files_for_experiment(24489 ,'YAF', EMOTIONS[2], pb)
    get_files_for_experiment(24493 ,'YAF', EMOTIONS[3], pb)
    get_files_for_experiment(24496 ,'YAF', EMOTIONS[4], pb)
    get_files_for_experiment(24500 ,'YAF', EMOTIONS[5], pb)
    get_files_for_experiment(24495 ,'YAF', EMOTIONS[6], pb)




# noe one made a 'download all link - for a collection of 16000 recordings - real smart'
def get_files_for_experiment(number ,subject, emotion, pb):
    base = BASE_URL + str(number) + '/'
    i = 0
    for word in WORDS:
        i += 1
        name =  '_'.join([subject, word, emotion]) + '.wav'
        print('name: ', name, ' number: ', number, ' subject: ', subject)
        url = base + str(i) + '/' + name
        print(i,'/',NUM_LINKS, url)
        # pb.add(1,[(name, 0)])
        get_file(url, DATA_DIR + name)





def add_permutaions(combinations, arr):
    return [combination + [option] for combination in combinations for option in arr]


def get_file(url, path):
    response = get(url) #,type='audio/x-wav')
    if 'text/html' in response.headers['content-type']:
        path += 'FAILED'

    with open(path, "wb") as file:
        file.write(response.content)

    # call(['curl', url, '>', path], shell=True)


if __name__ == '__main__':
    main()
