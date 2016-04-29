import pickle
import numpy as np
import h5py
import os
from scipy.io.wavfile import read
from libcontext import util
from util import mkdir_p

DATA_DIR = 'data/RAVDESS'
CORPORA_DIR = "corpora/RAVDESS"


mkdir_p(DATA_DIR)


f = h5py.File(DATA_DIR + "/waveforms.hdf5", "w")


# wavfiles = filter(lambda x: x.endswith('wav'), os.listdir('.'))
wavfiles = [filename for filename in os.listdir(CORPORA_DIR) if filename.endswith('wav')]


for wavfile in wavfiles:
    print 'extracting ' + wavfile
    waveform = read(CORPORA_DIR + '/' + wavfile)[1]

    dset = f.create_dataset(wavfile.split('.')[0], data=waveform)

f.close()
print 'COMPLETED WAVEFORM EXTRACTION'
