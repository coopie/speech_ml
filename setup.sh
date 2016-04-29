#bin/sh

# The setup for getting the copora

CORPORA_DIR=corpora


MAV_DIR=Montreal_Affective_Voices
MAV=http://vnl.psy.gla.ac.uk/sounds/Montreal_Affective_Voices.zip

RAVDESS_DIR=RAVDESS
RAVDESS=http://neuron.arts.ryerson.ca/ravdess/download.php?f=SpeechAO_AllActors.zip


function smartGet {
    local foldername=${1}
    local url=${2}

    if [ ! -d "$CORPORA_DIR/$foldername" ]; then
        echo "DOWNLOADING ${foldername}" &&
        mkdir -p $CORPORA_DIR/$foldername &&
        curl $url | c $CORPORA_DIR/$foldername
    fi
}

smartGet $MAV_DIR $MAV
smartGet $RAVDESS_DIR $RAVDESS
