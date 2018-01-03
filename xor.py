from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

ins = np.array([[0,0], [0,1], [1,0], [1,1]])
outs = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(8, input_dim=2))    # input layer
model.add(Activation('tanh'))

model.add(Dense(1))                 # output layer
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(ins, outs, batch_size=1, epochs=500)

print("0 ⊻ 0:", model.predict_proba(np.array([[0,0]])))
print("0 ⊻ 1:", model.predict_proba(np.array([[0,1]])))
print("1 ⊻ 0:", model.predict_proba(np.array([[1,0]])))
print("1 ⊻ 1:", model.predict_proba(np.array([[1,1]])))
