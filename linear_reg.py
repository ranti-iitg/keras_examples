import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Embedding, Input, merge, ELU
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2, activity_l2, l1
from keras.utils.np_utils import to_categorical
from keras import backend as K

#y=mx+b type regression
frame_in = Input(shape=(1,), name='data_input')
out = Dense(1, activation='linear', name='out', init='lecun_uniform')(frame_in)

model = Model(input=frame_in, output=out)
adam = Adam(lr=0.1)
model.compile(loss='mse',optimizer=adam)