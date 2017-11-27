import numpy as np


class ValueIteration:

	def __init__(self, env, grid, gamma=0.5):
		self.v_table = np.zeros(grid.shape)
		self.env = env
		self.grid = grid
		self.gamma = gamma
		self.epsilon = 1e-2
		self.actions = np.arange(4)

	def oneIteration(self):
		new_v_table = np.zeros(self.v_table.shape)
		delta = 0
		# For all states:
		for i in range(self.grid.shape[0]):
			for j in range(self.grid.shape[1]):
				terms = []
				for action in self.actions:
					s_ = self.env.get_action_state((i,j), action)
					terms.append((self.env.get_reward(s_[0], s_[1]) +  self.gamma * self.v_table[s_[0], s_[1]]) * ( 1 / float(len(self.actions))))
				a_optimal = max(terms)
				delta = max(delta, np.abs(a_optimal - self.v_table[i,j]))
				new_v_table[i,j] = a_optimal
		self.v_table = new_v_table
		return delta
		
	def runIterations(self):
		delta = np.inf
		while delta >= self.epsilon:
			delta = self.oneIteration()
			print delta
 