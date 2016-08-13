
class KerasGenerator(object):
    """Base class for a generator for keras. This class is used for autoencoders, where X is the same value as Y.
    """

    def __init__(self, data_source, batch_size=128):
        """TODO:."""
        self.data_source_x = data_source
        self.chunk_index = 0
        self.batch_size = batch_size

    def __next__(self):
        if self.chunk_index == len(self):
            self.chunk_index = 0

        x = self.data_source_x[self.chunk_index:self.chunk_index + self.batch_size]
        self.chunk_index += self.batch_size
        return (x, x)

    def __len__(self):
        """Return the size of the data, to the nearest `batch_size`"""
        return len(self.data_source_x) - (len(self.data_source_x) % self.batch_size)


class LabeledKerasGenerator(KerasGenerator):
    """Generator for labeled data."""

    def __init__(self, data_source_x, data_source_y, batch_size=128):
        assert len(data_source_x) == len(data_source_y)

        super().__init__(data_source_x, batch_size=batch_size)
        self.data_source_y = data_source_y

    def __next__(self):
        if self.chunk_index == len(self):
            self.chunk_index = 0

        x = self.data_source_x[self.chunk_index:self.chunk_index + self.batch_size]
        y = self.data_source_y[self.chunk_index:self.chunk_index + self.batch_size]
        self.chunk_index += self.batch_size
        return (x, y)
