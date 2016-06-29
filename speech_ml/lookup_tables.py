"""LookupTable is an adapter from one adressing space to another.

Typically this means changing from a uri adress space to a integer adress space.
"""

from abc import abstractmethod
import numpy as np
import random
from random import shuffle

random.seed(1337)


class LookupTable(object):
    """Base class for lookuptables"""

    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass



class TTVLookupTable(LookupTable):
    """Use a ttv split to create a lookup table.

    Also exposes slices of the lookup table corresponding to different data sets(i.e. test, train, validation sets).
    """
    def __init__(self, ttv, shuffle_in_set=False):
        """Create a table which rerturns"""
        self.__indices_for_sets = {}
        lookup_table = []

        index = 0
        for data_set in ['train', 'validation', 'test']:
            uris_for_set = []
            start_index = index

            subjects = ttv[data_set]
            for subjectID in subjects:
                uris = subjects[subjectID]
                for uri in uris:
                    index += 1
                    uris_for_set.append(uri)
            end_index = index
            self.__indices_for_sets[data_set] = (start_index, end_index)

            if shuffle_in_set:
                shuffle(uris_for_set)
            lookup_table += uris_for_set

        self.uris = np.array(lookup_table)


    def __getitem__(self, key):
        x = self.uris[key]
        if isinstance(x, np.str):
            return str(x)
        else:
            return [str(y) for y in x]


    def __len__(self):
        return len(self.uris)


    def get_set_bounds(self, set_name):
        """Return the slice index slice for a certain data set"""
        return self.__indices_for_sets[set_name]
