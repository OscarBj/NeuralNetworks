from scipy import signal, io, fftpack
import numpy as np
import matplotlib.pyplot as plt
import math

# Implementation of a neural network without keras or tf

# Sigmoid activation function
def sigmoid(x, derivative=False):
    y = 1/(1+np.exp(-x))
    if derivative == True:
        return y * (1-y)
    return y
	
# Heaviside activation function
def heaviside(x):
    return 1 if x>=0 else 0
	
def train2(X,C):
	b = np.random.normal(size=1) # that's the bias
	W = np.random.normal(size=1) # that's the weights
	n_epoch = 100
	n_error=0
	
	for e in range(n_epoch):
		n_error = 0
		for n in range(len(X)):
        # we compute the output y
		#x = X[n]
			x = X[n]
			#print(x)
			#x = np.expand_dims(x, axis=0) # a little trick here

			y = np.heaviside(np.dot(x, W)+b,0)
		#print(y)
        # we measure the error
			error = C[n] - y
		#print(error)
			#print(error)
        # we update the weights
			#W += X[n]*error
			W += X[n]*error
		#print(error)
		# we update the bias
			b += error
			if error != 0:
				n_error += 1
		print("epoch:", e, "percentage of error:", 100*n_error/len(X), "%")
	
	learned_C = []
	print(W)
	print(b)
	for n in range(len(X)):
		x = X[n]
		y = np.heaviside(np.dot(x, W)+b,0) 
		learned_C.append(y)
	
	learned_color = ["orange" if p>0 else "blue" for p in learned_C]
	plt.bar(np.arange(0,len(X),1), X,color=learned_color)
	plt.show()
	
#broken, TODO
def train(X,C):
	# params
	n_neurons = 2 
	# generate the weights and bias that will digest our inputs
	W_in = np.random.random((2,n_neurons)) 
	b_in = np.random.random(n_neurons)
	# generate the weights and bias that will spit the output 
	W_out = np.random.random((n_neurons,1))
	b_out = np.random.random(1)
	# normalize
	#ft = fftpack.fft(X)
	#ft = np.abs(ft)
	ft = X
	X_norm = (ft - np.mean(ft)) / np.std(ft)
	# [[a b][c d]]
	#print(X_norm)
	learning_rate = 1
	n_epoch = 100
	l_epoch = []
	Y = []
	for e in range(n_epoch):
		error_rate = 0
		for n in range(len(X)):
			x = X_norm[n]
			x = [x,n]
			x = np.expand_dims(x, axis=0) # a little trick here

			#[[a b]]
			#print(x)
        # we compute the outputs of the first layer
			l_in = np.dot(x, W_in)+b_in
        # and compute their activation 
			#h = sigmoid(l_in) 
			h = np.heaviside(l_in,0)
		# we compute the output s of the last layer
			l_out = np.dot(h, W_out)+b_out
        # and the activation of the output neuron (out output y)
			#y = sigmoid(l_out)
			y = np.heaviside(l_out,0)
			print(y)
			#print(y)
        # we compute the error
			error = C[n] - y
			error_rate += error**2
			#print(error)
        # we compute the delta for each layer
        # first the delta between 
			#y_delta = error * sigmoid(l_out, derivative=True) 
			y_delta = error * np.heaviside(l_out,0) 
			
			#h_delta = np.dot(y_delta,W_out.T) * sigmoid(l_in, derivative=True)
			h_delta = np.dot(y_delta,W_out.T) * np.heaviside(l_in,0)
			
			W_out += learning_rate * np.dot(h.T, y_delta)
			b_out += learning_rate * y_delta.squeeze()
			W_in += learning_rate * np.dot(x.T, h_delta)
			b_in += learning_rate * h_delta.squeeze()
		l_epoch.append(error_rate.squeeze()/len(X))
		#clear_output(wait = True)
		
		print("epoch:", e, "error:", error_rate/len(X))
    #plt.pause(0.5)
	#plt.plot
	print(W_in) # [[0.90082542 0.56672738] [0.83632847 0.48059755]]
	print(b_in) # [0.83437138 0.270109]
	print(W_out)# [[-1.72666539] [-1.96150462]]
	print(b_out)# [-1.64411228]
	learned_C = []
	for n in range(len(X)):
		# look into the treshold of 0.5 and what it means
		x = X_norm[n]
		#h = sigmoid(np.dot(x, W_in)+b_in)
		h = np.heaviside(np.dot(x, W_in)+b_in,0)
		
		#y = sigmoid(np.dot(h, W_out)+b_out)
		y = np.heaviside(np.dot(h, W_out)+b_out,0)
		
		learned_C.append(y.flatten()[0])
	#print(learned_C)	
	learned_color = ["orange" if p>0 else "blue" for p in learned_C]
	
	#o = []
	plt.bar(np.arange(0,len(X),1), X,color=learned_color)
	plt.show()
	
def getTopNfrequencies(fs,signal,N):
	ft = fftpack.fft(signal)
	N = -1*N
	l = len(signal)
	x = np.abs(ft)
	
	m = []
	for i in np.argpartition(x[0:math.floor(l/2)], N)[N:]:
		 m.append(i*fs/l)
		 #print("sample: "+str(i)+ " amp: "+str(x[i]))
	#plt.plot(x)
	#plt.show()
	return m

def createLabels(fs, signal, signal_ft):
	M = []
	l = len(signal)
	C = getTopNfrequencies(fs,signal,5)
	#print(C)
	isOk = False
	for i in range(0,len(signal_ft)):
		isOk = False
		for c in C:
			# treshold
			if(np.abs(c-(i*fs/l))==0):
				isOk = True
		
		if(isOk):
			M.append(1)
		else:
			M.append(0)
	return M
	
def infere(X, Win,bin,Wout,bout):
	X_norm = (X - np.mean(X)) / np.std(X)
	learned_C = []
	for n in range(len(X)):
		# look into the treshold of 0.5 and what it means
		x = X_norm[n]
		h = np.heaviside(np.dot(x, Win)+bin,0)
		y = np.heaviside(np.dot(h, Wout)+bout,0)
		print(y)
		#learned_C.append(y.flatten()[1])
	#print(learned_C)	
	learned_color = ["orange" if p>0 else "blue" for p in learned_C]
	plt.bar(np.arange(0,1000,1), X,color=learned_color)
	plt.show()

def infere2(X,w,b):
	learned_C = []
	for n in range(len(X)):
		x = X[n]
		y = np.heaviside(np.dot(x, w)+b,0) 
		learned_C.append(y)
	
	learned_color = ["orange" if p>0 else "blue" for p in learned_C]
	plt.bar(np.arange(0,len(X),1), X,color=learned_color)
	plt.show()

def getSine(Fs,frq,l):
	ts = 1/Fs
	t = l/Fs
	x = np.arange(0,t,ts)
	y = np.sin(2*np.pi*frq*x)
	return y
	
	
def main():
	# load data
	fs = 16000
	inp = io.loadmat("inp.mat")
	ref = io.loadmat("ref.mat") # used for training
	s1 = inp['y'].flatten()
	
	s2 = ref['signal'].flatten() # used for training
	plt.plot(s2)
	plt.plot(s1)
	plt.show()
	
	s2 = s2[69000:71000]
	s1 = s1[69000:71000]
	
	s3 = getSine(fs,40,2000) + getSine(fs,50,2000) + getSine(fs,78,2000) + getSine(fs,9,2000)
	ft = fftpack.fft(s1)
	ft = np.abs(ft)
	#print(s2)
	#print(ft)
	#plt.plot(ft)
	#plt.show()
	
	#C = createLabels(fs,s2,ft)
	ft = ft[:1000]
	
	#print(s2)
	#print(ref)
	#plt.bar(np.arange(0,1000,1),ft,0.4,color="orange")
	#plt.show()
	#train2(ft,C)
	infere2(ft,4.16696126,-14.44260816)
	#win = [[0.40511925, 0.5972272 ], [0.12258102, 0.07323874]]
	#bin = [-0.20020464,  0.19401821]
	#wout = [[-0.656373  ], [ 0.40482149]]
	#bout = [-0.895025]
	#infere(ft,win,bin,wout,bout)

	#learned_color = ["orange" if p>0 else "blue" for p in C]
	#plt.bar(np.arange(0,2000,1), ft,color=learned_color)
	#plt.bar(np.arange(0,1000,1), ft,color=learned_color)
	
	#plt.plot(ft)
	#plt.show()

main()


#with np.load('sd.npz') as data:
#    X = data['X']
#    C = data['C']
#train(X,C)	
#print(C)
#plt.plot(X)
#plt.show()