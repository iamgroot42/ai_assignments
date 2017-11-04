import numpy as np
np.random.seed(42)

from keras.datasets import mnist
from keras.utils import np_utils

from Layer import Layer
from Activation import Sigmoid, SoftMax, ReLU
from Model import Model
from Error import Error


def process_data():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(60000, 784)
	y_train = y_train.reshape(60000, 1)
	X_test = X_test.reshape(10000, 784)
	y_test = y_test.reshape(10000, 1)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	X_test = np.matrix(X_test)
	y_test = np.matrix(y_test)
	return (X_train, y_train), (X_test, y_test)


def single_layer(X_train, X_test, y_train, y_test, verbose=False):
	m = Model(Error())
	m.add_layer(Layer((784,10), SoftMax(), 'fanin'))
	t_acc = (1-m.train(X_train, y_train, verbose)) * 100
	print "Train accuracy", t_acc, "%"
	print "Test accuracy", (1-m.test(X_test, y_test)) * 100, "%"


def multi_layer(X_train, X_test, y_train, y_test, verbose=False):
	batch_size = 128
	learning_rate = 1e-2
	momentum_rate = 0.25
	m = Model(Error(), learning_rate, momentum_rate, batch_size)
	m.add_layer(Layer((784,100), ReLU(), 'fanin'))
	# m.add_layer(Layer((784,100), ReLU()))
	m.add_layer(Layer((100,10), SoftMax(), 'fanin'))
	t_acc = (1-m.train(X_train, y_train, verbose)) * 100
	print "Train accuracy", t_acc, "%"
	print "Test accuracy", (1-m.test(X_test, y_test)) * 100, "%"


if __name__ == "__main__":
	(X_train, y_train), (X_test, y_test) = process_data()
	#multi_layer(X_train, X_test, y_train, y_test, False)
	single_layer(X_train, X_test, y_train, y_test, True)
