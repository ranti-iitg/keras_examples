from linear_reg import model
import numpy as np
from tqdm import tqdm
from random import random

x = np.zeros(10)
y = np.zeros(10)

for i in range(10):
	y[i]=3+2*i+random()
	x[i]=i

num_steps=10
h = model.fit(x,y, verbose=1, validation_split=0.1, nb_epoch=100,shuffle=True)
print model.get_weights()
