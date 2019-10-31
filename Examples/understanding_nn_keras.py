from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import losses

from scipy import signal, io, fftpack
import numpy as np
import matplotlib.pyplot as plt
import math

# Function for creating a complex dataset
def vcd_generator(n=[1000, 1000, 1000, 1000]):
    n_blue1 = n[0]
    n_blue2 = n[1]
    blue_u1 = (5, 5)
    blue_u2 = (0, 0)
    blue_s1 = [[1.5, 0], [0, 1.5]]
    blue_s2 = [[1.5, 0], [0, 1.5]]
    a = np.random.multivariate_normal(blue_u1, blue_s1, (n_blue1))
    a = np.array([x  for x in a if (2.5 <= x[0] < 7.5) and (2.5 <= x[1] < 7.5) ])
    b = np.random.multivariate_normal(blue_u2, blue_s2, (n_blue2))
    b = np.array([x  for x in b if (-2.5 <= x[0] < 2.5) and (-2.5 <= x[1] < 2.5) ])
    
    x_blue = np.concatenate( ( a , b ) , axis=0 )
    y_blue = np.zeros(len(x_blue))

    n_orange1 = n[2]
    n_orange2 = n[3]
    orange_u1 = (5, 0)
    orange_u2 = (0, 5)
    orange_s1 = [[1.5, 0], [0, 1.5]]
    orange_s2 = [[1.5, 0], [0, 1.5]]
    c = np.random.multivariate_normal(orange_u1, orange_s1, (n_orange1))
    c = np.array([x  for x in c if (2.5 <= x[0] < 7.5) and (-2.5 <= x[1] < 2.5) ])
    d = np.random.multivariate_normal(orange_u2, orange_s2, (n_orange2))
    d = np.array([x  for x in d if (-2.5 <= x[0] < 2.5) and (2.5 <= x[1] < 7.5) ])
    x_orange = np.concatenate( ( c, d ), axis=0)
    y_orange = np.ones(len(x_orange))

    X = np.concatenate((x_blue, x_orange),axis=0)
    #X = (X - np.mean(X)) / np.std(X)
    Y = np.concatenate((y_blue, y_orange), axis=0)

    ## Lets randomize a bit ...
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]

    ## resizing of Y to fit with our network 
    Y = np.expand_dims(Y, axis=1)
    return X, Y
	
# Function for generating a complex dataset 
def cd_generator(n=[1000, 1000, 1000, 1000]):
    n_blue1 = n[0]
    n_blue2 = n[1]
    blue_u1 = (10, 10)
    blue_u2 = (0, 0)
    blue_s1 = [[1, 0], [0, 1]]
    blue_s2 = [[1, 0], [0, 1]]
    x_blue = np.concatenate( ( np.random.multivariate_normal(blue_u1, blue_s1, (n_blue1)), np.random.multivariate_normal(blue_u2, blue_s2, (n_blue2))), axis=0)
    y_blue = np.zeros(len(x_blue))

    n_orange1 = n[2]
    n_orange2 = n[3]
    orange_u1 = (10, 0)
    orange_u2 = (0, 10)
    orange_s1 = [[1, 0], [0, 1]]
    orange_s2 = [[1, 0], [0, 1]]
    x_orange = np.concatenate( ( np.random.multivariate_normal(orange_u1, orange_s1, (n_orange1)), np.random.multivariate_normal(orange_u2, orange_s2, (n_orange2))), axis=0)
    y_orange = np.ones(len(x_orange))

    X = np.concatenate((x_blue, x_orange),axis=0)
    #X = (X - np.mean(X)) / np.std(X)
    Y = np.concatenate((y_blue, y_orange), axis=0)

    ## Lets randomize a bit ...
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]

    ## resizing of Y to fit with our network 
    Y = np.expand_dims(Y, axis=1)
    return X, Y
	
# our training dataset
X, C = vcd_generator()
#plt.scatter(X[:,0], X[:,1], c=['tab:orange' if p>=0.5 else "tab:blue" for p in C ])
#plt.show()

# our test dataset 
Xnew, Cnew = vcd_generator(n=[10,10,10,10])
#plt.scatter(Xnew[:,0], Xnew[:,1], c="black")
#plt.show()

X_norm = (X - np.mean(X)) / np.std(X)
Xnew_norm = (Xnew - np.mean(Xnew)) / np.std(Xnew)

#	KERAS
model = Sequential() # create sequential model
opt = optimizers.SGD(lr=0.001) # Optimization algorithm, minimizing error using (back) propagation to previous layer correcting their W and B
#ls = losses.binary_crossentropy() # Loss function, error calculation technique
#ls = losses.sparse_categorical_crossentropy(C, X, from_logits=False)

# Input layer + 1 hidden layer
model.add(Dense(5, input_dim=2, activation='relu')) # input layer maps to (hidden) layer with 10 neurons/nodes, act funct = relu 

# Hidden layers
#model.add(Dense(7, activation='sigmoid')) # input layer maps to (hidden) layer with 10 neurons/nodes, act funct = relu 
model.add(Dense(3, activation='sigmoid')) # input layer maps to (hidden) layer with 10 neurons/nodes, act funct = relu 

# We use one hot encoding -> labels are vectors with the length of number of classes -> [X X]
# Output layer
model.add(Dense(2,activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy']) # loss = binary (sound recognized or not
#model.compile(loss=ls, optimizer=opt, metrics=['accuracy']) # loss = binary (sound recognized or not

# Use 20% of the training data for validation
model.fit(X, C, validation_split=0.20, epochs=5, batch_size=2, shuffle=True) # batch size = nr times each sample is iterated before model is updated, epoch = dataset iterations
				# validation set = [(sample,label),(sample,label),...]
#model.fit(X, C, validation_data=valid_set, epochs=5, batch_size=2, shuffle=True) # batch size = nr times each sample is iterated before model is updated, epoch = dataset iterations

print(model.summary())
predictions = model.predict(X, batch_size=10)
#predictions = model.predict_classes(X)
pred_color = ["tab:orange" if p[0]>=0.5 else "tab:blue" for p in predictions]

predictions2 = model.predict(Xnew, batch_size=10)
#predictions2 = model.predict_classes(Xnew)
pred_color2 = ["tab:orange" if p[0]>=0.5 else "tab:blue" for p in predictions2]

plt.scatter(Xnew[:,0], Xnew[:,1] ,c=pred_color2)
plt.scatter(X[:,0], X[:,1], marker='x' ,c=pred_color, alpha=0.1)
plt.show()
