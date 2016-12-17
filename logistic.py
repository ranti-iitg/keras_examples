from logistic_reg import model
import numpy as np
from tqdm import tqdm
from random import random

x = np.zeros((10,2))
y = np.zeros((10,2))

for i in range(10):
	temp=2*i+random()
	x[i][0]=i
	x[i][1]=temp
	if temp>2*i+0.5:
		y[i][1]=1
	else:
		y[i][0]=1



h = model.fit(x,y, verbose=1, validation_split=0.1, nb_epoch=1000,shuffle=True)
print model.get_weights()
