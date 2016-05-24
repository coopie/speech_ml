# functions to deterministically make test, train, validate sets for datasets

import sys
import os
import yaml
import random
import numpy as np
import posixpath

DEFAULT_TTV_RATIO = (20, 60, 20)

def main():
    make_ttv_yaml(sys.argv[1:-1], sys.argv[-1], deterministic=False)


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
    data_sets = make_ttv(dataset, ttv_ratio=ttv_ratio, deterministic=deterministic)

    def get_for_ttv(key):
        return (
            data_sets['test'][key],
            data_sets['train'][key],
            data_sets['validation'][key]
        )

    test, train, validation = get_for_ttv('paths')

    number_of_files_for_each_set  = list(get_for_ttv('number_of_files'))

    number_of_subjects_for_each_set = [len(x) for x in get_for_ttv('subjects')]

    dict_for_yaml = {
        'split': number_of_files_for_each_set,
        'subject_split': number_of_subjects_for_each_set,
        "test": test,
        "train": train,
        "validation": validation
    }

    with open(path_to_ttv_file, 'w') as f:
        yaml.dump(dict_for_yaml, f, default_flow_style=False)

def make_ttv(dataset, ttv_ratio=DEFAULT_TTV_RATIO, deterministic=False):
    """
    Returns a dict of test,train,validation sets with information regarding the split (lists of paths to data).
    Currently only separates by subjectID.

    Prioitises having a diverse set than a well fitting set. This means that the
    'knapsacking' is done by subjects with the least videos first, so that each set can have
    as many different subjects in as possible.

    returns {
        'subjects': list of subjectIDs in the set
        'number_of_files': number of files the set has
        'expected_size' : (ttv_ratio[2] * number_of_resources)
        'paths': list of paths to the files
    }
    """

    sizes_and_ids = [(len(dataset[key]), key) for key in dataset]
    number_of_resources = sum([x[0] for x in sizes_and_ids])


    # Get the smallest first
    sizes_and_ids.sort()
    if not deterministic:
        random.shuffle(sizes_and_ids)

    # normalise ttv_ratio
    ttv_ratio = [x/sum(ttv_ratio) for x in ttv_ratio]

    data_sets = {}

    for key, ratio in zip(['test', 'train', 'validation'], ttv_ratio):
        data_sets[key] = {
            'subjects': [],
            'number_of_files': 0,
            'expected_size' : (ratio * number_of_resources)
        }



    last_trip = False
    set_names = list(data_sets.keys())
    i = 0
    while len(sizes_and_ids) > 0:
        i += 1
        s = data_sets[set_names[i % len(set_names)]]
        if s['number_of_files'] < s['expected_size']:
            size, subjectID = sizes_and_ids.pop(0)
            s['subjects'].append(subjectID)
            s['number_of_files'] += size

    def get_filenames(subjects):
        return sum(list(map(lambda x: dataset[x], subjects)), [])

    # turn subjectIDs into paths to files
    for data_set in data_sets:
        s = data_sets[data_set]
        s['paths'] = get_filenames(s['subjects'])

    return data_sets


def split_list(arr, proportion):
    split_index = int(len(arr)*proportion)
    return arr[:split_index], arr[split_index:]


def get_dataset(corpora):
    """
    Returns a dictionary of subjectID -> [path_to_wav_file]
    """
    # TODO: make filter methods for the files

    def make_posix_path(dirpath, filename):
        dirpath = dirpath.split(os.sep).join(posixpath.sep)
        return posixpath.join(dirpath, filename)

    wav_files_in_corpora = filter(lambda x: x.endswith('.wav'),
        sum(
            [list(map(lambda x: make_posix_path(corpus, x), os.listdir(corpus))) for corpus in corpora],
            []
        ),
    )

    dataset = {}
    for wav_file in wav_files_in_corpora:
        subjectID = os.path.split(wav_file)[-1].split('.')[0].split('_')[0]

        if subjectID in dataset:
            dataset[subjectID].append(wav_file)
        else:
            dataset[subjectID] = [wav_file]

    return dataset

if __name__ == '__main__':
    main()
