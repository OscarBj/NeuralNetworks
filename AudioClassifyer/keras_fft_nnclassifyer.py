from keras.models import Sequential
from keras.layers import Dense

from scipy import signal, io, fftpack
import numpy as np
import matplotlib.pyplot as plt
import math

def getSine(Fs,frq,l):
	ts = 1/Fs
	t = l/Fs
	x = np.arange(0,t,ts)
	y = np.sin(2*np.pi*frq*x)
	return y
	

fs=16000
inp = io.loadmat("inp.mat")
ref = io.loadmat("ref.mat")
s2 = ref['signal'].flatten() 	# used for training
s1 = inp['y'].flatten() 		# used for testing

ref_signal = s2[69000:71000]
ft = fftpack.fft(ref_signal)
ft = np.abs(ft)
ft = ft[:1000]
c1 = [1]

test_noise = s1[5000:7000]
ft_noise = fftpack.fft(test_noise)
ft_noise = np.abs(ft_noise)
ft_noise = ft_noise[:1000]
c3 = [0]

test_ref = s1[69000:71000]
ft_testref = fftpack.fft(test_ref)
ft_testref = np.abs(ft_testref)
ft_testref = ft_testref[:1000]

gs1 = 0
for i in range(100,300):
	gs1+=getSine(fs,i,2000)
	
ft_gs1 = fftpack.fft(gs1)
ft_gs1 = np.abs(ft_gs1)
ft_gs1 = ft_gs1[:1000]
c2 = [0]

gs2 = 0
for i in range(600,800):
	gs2+=getSine(fs,i,2000)
	
ft_test = fftpack.fft(gs2)
ft_test = np.abs(ft_test)
ft_test = ft_test[:1000]

#C = [c1,c2]
C = np.array([c1,c2, c3])
C.reshape(3,1)
X = np.array([ft.tolist(),ft_gs1.tolist(),ft_noise.tolist()])
X.reshape(3,1000)
#X = np.concatenate([ft,ft_gs1])
print(X)
model = Sequential() # create sequential model
					# input_dim is the abstraction of input layer
model.add(Dense(10, input_dim=1000, activation='relu')) # input layer AND hidden layer with 10 neurons/nodes, act funct = relu 
model.add(Dense(5, activation='relu')) # next layer maps to 5 neurons/nodes, same act funct
model.add(Dense(1, activation='sigmoid')) # output layer is binary, hence 1 neuron/node
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy']) # loss = binary (sound recognized or not
# Train Model
model.fit(X, C, epochs=100, batch_size=10) # batch size = nr times each sample is iterated before model is updated, epoch = dataset iterations
_, accuracy = model.evaluate(X, C) # verify
print('Accuracy: %.2f' % (accuracy*100))

# Predict
# Predict against training data
predictions = model.predict_classes(ft.reshape(1,1000))
predictions2 = model.predict_classes(ft_gs1.reshape(1,1000))
predictions3 = model.predict_classes(ft_noise.reshape(1,1000))
# Predict against test data
predictions4 = model.predict_classes(ft_testref.reshape(1,1000))

learned_color1 = ["orange" if p>0 else "blue" for p in predictions]
learned_color2 = ["orange" if p>0 else "blue" for p in predictions2]
learned_color3 = ["orange" if p>0 else "blue" for p in predictions3]
learned_color4 = ["orange" if p>0 else "blue" for p in predictions4]

fig,subs = plt.subplots(2,3)
			#row,col,idx
subs[0, 0].set_title('training data - true')
subs[0, 0].bar(np.arange(0,len(ft),1), ft,color=learned_color1)

subs[0, 1].set_title('training data - false')
subs[0, 1].bar(np.arange(0,len(ft_gs1),1), ft_gs1,color=learned_color2)

subs[0, 2].set_title('training data - false')
subs[0, 2].bar(np.arange(0,len(ft_noise),1), ft_noise,color=learned_color3)

subs[1, 0].set_title('test data - true')
subs[1, 0].bar(np.arange(0,len(ft_testref),1), ft_testref,color=learned_color4)

plt.show()

# Display prediction results
# Verify training data
#learned_color = ["orange" if p>0 else "blue" for p in predictions]
#plt.bar(np.arange(0,len(ft),1), ft,color=learned_color)
#plt.show()


#learned_color = ["orange" if p>0 else "blue" for p in predictions2]
#plt.bar(np.arange(0,len(ft_gs1),1), ft_gs1,color=learned_color)
#plt.show()


#learned_color = ["orange" if p>0 else "blue" for p in predictions3]
#plt.bar(np.arange(0,len(ft_noise),1), ft_noise,color=learned_color)
#plt.show()

# Check test data
#learned_color = ["orange" if p>0 else "blue" for p in predictions4]
#plt.bar(np.arange(0,len(ft_testref),1), ft_testref,color=learned_color)
#plt.show()