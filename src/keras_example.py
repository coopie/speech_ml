from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np

inputs = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
outputs = np.array([0,1,1,0]).astype('float32')

xor = Sequential()
xor.add(Dense(4, activation="sigmoid", input_shape=(2,)))
xor.add(Dense(1, activation="sigmoid"))

xor.compile(loss="mean_absolute_error", optimizer="rmsprop")

xor.fit(inputs, outputs, nb_epoch=10000)

print "Model loss: %4f" % xor.evaluate(inputs, outputs)

print "XOR(0,0) ~= %4f" % xor.predict(inputs[0,:].reshape(1,2))
print "XOR(0,1) ~= %4f" % xor.predict(inputs[1,:].reshape(1,2))
print "XOR(1,0) ~= %4f" % xor.predict(inputs[2,:].reshape(1,2))
print "XOR(1,1) ~= %4f" % xor.predict(inputs[3,:].reshape(1,2))
