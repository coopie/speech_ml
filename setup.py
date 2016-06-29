from setuptools import setup


setup(
    name='speech_ml',
    version='0.0.1',
    description='Deep Learning utilities for audio using keras',
    author='Sam Coope',
    author_email='sam.j.coope@gmail.com',
    url='https://github.com/coopie/speech_ml',
    download_url='https://github.com/coopie/speech_ml/archive/master.zip',
    license='MIT',
    install_requires=['keras', 'pyyaml', 'sklearn', 'docopt', 'tqdm', 'h5py'],
    packages=['speech_ml']
)
