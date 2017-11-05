import numpy as np
import random


class Layer():
	def __init__(self, weight_shape, activation, init='random'):
		fan_in = weight_shape[0]
		fan_out = weight_shape[1]
		if init == 'zero':
			print('Zeros')
			self.weights = np.zeros((weight_shape[0], weight_shape[1]))
		elif init == 'random':
			print('Random')
			self.weights = np.random.rand(weight_shape[0], weight_shape[1]) / 100
		elif init == 'fanin':
			print('Fan-in')
			self.weights = np.random.uniform(-np.sqrt(3/fan_in), np.sqrt(3/fan_in), (weight_shape[0], weight_shape[1]))
		elif init == 'fanout':
			print('Fan-out')
			self.weights = np.random.uniform(-np.sqrt(3/fan_out), np.sqrt(3/fan_out), (weight_shape[0], weight_shape[1]))
		elif init =='faninout':
			print('Fan in-out')
			self.weights = np.random.uniform(-np.sqrt(6/(fan_in+fan_out)), np.sqrt(6/(fan_in+fan_out)), (weight_shape[0], weight_shape[1]))
		self.output = np.zeros(weight_shape)
		self.gradient = None
		self.momentum = 0.0
		self.activation = activation

	def forward(self, input_data):
		self.output = input_data * self.weights
		return self.activation.func(self.output)
