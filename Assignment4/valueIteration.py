import numpy as np


class ValueIteration:

	def __init__(self, env, grid, gamma=0.5):
		self.v_table = np.zeros(grid.shape)
		self.env = env
		self.grid = grid
		self.gamma = 0.5
		self.epsilon = 1e-4
		self.actions = np.arange(4)

	def oneIteration(self):
		delta = 0
		new_v = np.copy(self.v_table)
		# For all states:
		for i in range(self.grid.shape[0]):
			for j in range(self.grid.shape[1]):
				terms = []
				for action in self.actions:
					s_ = self.env.get_action_state((i,j), action)
					terms.append((self.env.get_reward(s_) +  self.gamma * self.v_table[s_[1], s_[0]]) * ( 1 / float(len(self.actions))))
				a_optimal = max(terms)
				delta = max(delta, np.abs(a_optimal - self.v_table[i,j]))
				new_v[i,j] = a_optimal
		self.v_table = np.copy(new_v)
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
				s_ = env.get_action_state((ploc[1],ploc[0]), action)
				terms.append(self.v_table[s_[1], s_[0]])
			best_action = np.argmax(terms)
 			ploc, _, reward, game_over = env.frame_step(best_action)
 			steps += 1
 			total_reward += reward
 			print ploc
 		return steps, total_reward