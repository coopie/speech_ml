
class KerasGenerator(object):
    """Base class for a generator for keras. This class is used for autoencoders, where X is the same value as Y.
    """

    def __init__(self, data_source, batch_size=128):
        """TODO:."""
        self.data_source = data_source
        self.chunk_index = 0
        self.batch_size = batch_size

    def __next__(self):
        if self.chunk_index == len(self):
            self.chunk_index = 0

        x = self.data_source[self.chunk_index:self.chunk_index + self.batch_size]
        self.chunk_index += self.batch_size
        return (x, x)

    def __len__(self):
        return len(self.data_source) - (len(self.data_source) % self.batch_size)
