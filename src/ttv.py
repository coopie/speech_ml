# functions to deterministically make test, train, validate sets for datasets

import sys
import os
import yaml
import random
import numpy as np

DEFAULT_TTV_RATIO = (20, 60, 20)

def main():
    make_ttv_yaml(sys.argv[1:-1], sys.argv[-1])


def make_ttv_yaml(corpora, path_to_ttv_file, ttv_ratio=DEFAULT_TTV_RATIO, deterministic=False):
    """ Creates a test, train, validation from the corpora given and saves it as a YAML filename.
        Each set will be subject independent, meaning that no one subject can have data in more than one
        set

    # Arguments;
        corpora: a list of the paths to corpora used (these have to be formatted accoring to notes.md)
        path_to_ttv_file: the path to where the YAML file be be saved
        ttv_ratio: a tuple (e.g. (1,4,4) of the relative sizoe of each set)
        deterministic: whether or not to shuffle the resources around when making the set
    """
    dataset = get_dataset(corpora)
    test, train, validation = make_ttv(dataset, ttv_ratio=ttv_ratio, deterministic=deterministic)

    dict_for_yaml = {
        "test": test,
        "train": train,
        "validation": validation
    }

    with open(path_to_ttv_file, 'w') as f:
        yaml.dump(dict_for_yaml, f, default_flow_style=False)

def make_ttv(dataset, ttv_ratio=DEFAULT_TTV_RATIO, deterministic=False):
    """
    Returns a tuple of test,train,validation sets (lists of paths to data).
    Currently only separates by subjectID.
    """

    total_ttv = sum(ttv_ratio)

    subjects = dataset.keys()
    if not deterministic:
        random.shuffle(subjects)

    num_subjects = len(subjects)

    test_subjects, rest = split_list(subjects, ttv_ratio[0]/float(sum(ttv_ratio)))
    train_subjects, rest = split_list(rest, ttv_ratio[1]/float(sum(ttv_ratio[1:])))
    validation_subjects = rest


    def get_filenames(subjects):
        return sum(map(lambda x: dataset[x], subjects), [])

    test_data, train_data, validation_data = \
        map(get_filenames, (test_subjects, train_subjects, validation_subjects))

    return (test_data, train_data, validation_data)


def split_list(arr, proportion):
    split_index = int(len(arr)*proportion)
    return arr[:split_index], arr[split_index:]


def get_dataset(corpora):
    """
    Returns a dictionary of subjectID -> [path_to_wav_file]
    """
    # TODO: make filter methods for the files

    wav_files_in_corpora = filter(lambda x: x.endswith('.wav'),
        sum(
            [map(lambda x: corpus + '/' + x, os.listdir(corpus)) for corpus in corpora],
            []
        ),
    )

    dataset = {}
    for wav_file in wav_files_in_corpora:
        subjectID = wav_file.split('/')[-1].split('.')[0].split('_')[0]

        if subjectID in dataset:
            dataset[subjectID].append(wav_file)
        else:
            dataset[subjectID] = [wav_file]

    return dataset

if __name__ == '__main__':
    main()
