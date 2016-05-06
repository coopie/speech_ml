# speech-ml <img src="https://travis-ci.org/coopie/speech-ml.svg" alt="build:started">
A repo to test out detecting stress/emotional state of people in recorded voice

### Requirements

### Aims

* [x] Find some corpora to start my research on.

* [ ] Get a baseline approach to the vocalizations corpus and set up a good workflow.

* [ ] Establish some methods to try with the MAV corpus (spectrograms, waveform, etc.)

* [ ] Train the models with the different methods, record the results.

* [ ] Publish findings.


## Tests

Tests are labeled by a filename `"*_test.py"`

TODO:

* make init script nicer for someone else to use(in the far future)
* make the fpython stuff nicer to use (currently in the bin folder of env, setup needs to add it to the folder)
* Dockerise so others can easily develop (no real way of testing with scientific python packages with travis at the moment, unless docker is used)
* cache the ttv data_sets
* better logging for training
* tests for learning.py - v important
