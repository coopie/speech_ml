# Used for resampling audio to 16KHz
# For this to work, ffmpeg must be avaliable from the command line
from subprocess import call


def main():
    print('not done yet')


def downsample(path, verbose=True):
    verbose_level = 'info' if verbose else 'panic'
    call(['ffmpeg', '-loglevel', verbose_level, '-y', '-i', path, '-ar', '16000', path])


if __name__ == '__main__':
    main()
