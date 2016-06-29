import numpy as np


def pad_or_slice(waveform, width):
    """
    Either takes the middle slice of the waveform (if waveform is longer)
    or randomly pads the waveform on either side
    """
    if len(waveform) > width:
        return middle_slice(waveform, width)
    else:
        new_waveform = np.zeros(width, dtype='int16')
        index = int(np.random.rand(1) * (width - len(waveform)))
        new_waveform[index:index + len(waveform)] = waveform
        return new_waveform


def middle_slice(waveform, width):
    middle = len(waveform) // 2
    top = middle + (width // 2) + width % 2
    bottom = middle - (width // 2)
    return waveform[bottom:top]
