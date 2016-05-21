import os

import src_context
from resampling import downsample

DIR = 'corpora/toronto/'

def main():
    for filename in os.listdir(DIR):
        if filename.endswith('FAILED'):
            os.remove(DIR + filename)
        else:
            downsample(DIR + filename, verbose=False)


if __name__ == '__main__':
    main()
