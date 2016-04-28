#bin/sh

# The setup for getting the copora

CORPORA_DIR=corpora
VOCALIZATIONS_DIR=Vocalizations

VOCALIZATIONS=http://vnl.psy.gla.ac.uk/sounds/Montreal_Affective_Voices.zip



if [ ! -d "$CORPORA_DIR" ]; then
  mkdir -p $CORPORA_DIR/$VOCALIZATIONS_DIR
fi &&
curl $VOCALIZATIONS | tar -xf- -C $CORPORA_DIR/$VOCALIZATIONS_DIR
