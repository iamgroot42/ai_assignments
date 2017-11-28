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
		delta = 0
		# For all states:
		for i in range(self.grid.shape[0]):
			for j in range(self.grid.shape[1]):
				terms = []
				for action in self.actions:
					s_ = self.env.get_action_state((i,j), action)
					terms.append((self.env.get_reward(s_) +  self.gamma * self.v_table[s_[0], s_[1]]) * ( 1 / float(len(self.actions))))
				a_optimal = max(terms)
				delta = max(delta, np.abs(a_optimal - self.v_table[i,j]))
				self.v_table[i,j] = a_optimal
		return delta
		
	def runIterations(self):
		delta = np.inf
		while delta >= self.epsilon:
			delta = self.oneIteration()
			print delta
 
 	def run_agent(self, ploc, env):
 		game_over = False
 		total_reward = 0
 		steps = 0
 		while not game_over:
 			terms = []
 			for action in self.actions:
				s_ = env.get_action_state((ploc[0],ploc[1]), action)
				terms.append(self.v_table[s_[0], s_[1]])
			best_action = np.argmax(terms)
 			ploc, _, reward, game_over = env.frame_step(best_action)
 			steps += 1
 			print ploc
 			# print env.grid
 			total_reward += reward
 		return steps, total_reward