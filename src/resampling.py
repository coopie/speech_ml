# Used for resampling audio to 16KHz
# For this to work, ffmpeg must be avaliable from the command line
from subprocess import call
from keras.utils.generic_utils import Progbar

from sys import argv
import os

def main():
    corpora = argv[1:]
    num_files = sum([len(os.listdir(x)) for x in corpora])
    pb = Progbar(num_files)
    for corpus in corpora:
        for filename in os.listdir(corpus):
            if filename.endswith('wav'):
                downsample(os.path.join(corpus, filename), verbose=False)
            pb.add(1)


def downsample(path, verbose=True):
    verbose_level = 'info' if verbose else 'panic'
    call(['ffmpeg', '-loglevel', verbose_level, '-y', '-i', path, '-ar', '16000', path])


if __name__ == '__main__':
    main()
