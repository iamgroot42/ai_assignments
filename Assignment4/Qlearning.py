import numpy as np
from tqdm import tqdm


class Qlearning:

	def __init__(self, env, grid, gamma=0.5, learning_rate=0.5, nb_episodes=1000):
		self.Q_table = np.zeros(grid.shape + (4,))
		self.env = env
		self.grid = grid
		self.gamma = 0.5
		self.epsilon = 0.2
		self.learning_rate = learning_rate
		self.actions = np.arange(4)
		self.nb_episodes = nb_episodes
		self.max_steps = 100

	def runIterations(self):
		rewards = []
		for i in tqdm(range(self.nb_episodes)):
			episode_reward = 0
			game_over = False
			for _ in range(self.max_steps):
				if game_over:
					break
				best_action = None
				ploc = self.env.ploc
				y, x = ploc
				# print self.env.ploc
				if np.random.rand() < self.epsilon:
					best_action = np.random.choice(self.actions, 1)[0]
				else:
					best_action = np.argmax(self.Q_table[ploc[1], ploc[0], :])
				new_ploc, _, reward, game_over = self.env.frame_step(best_action)
				best_action_ = np.argmax(self.Q_table[new_ploc[1], new_ploc[0], :])
				episode_reward += reward
				# print reward
				self.Q_table[x, y, best_action] += self.learning_rate * (reward
							+ self.gamma * self.Q_table[new_ploc[1], new_ploc[0], best_action_] - self.Q_table[x, y, best_action])
				ploc = new_ploc
			rewards.append(episode_reward)
			self.env.resetEnv()
			# return rewards
		return rewards

 	def run_agent(self, ploc, env):
 		game_over = False
 		total_reward = 0
 		steps = 0
 		while not game_over:
 			print env.ploc
			best_action = np.argmax(self.Q_table[ploc[1], ploc[0], :])
 			ploc, grid, reward, game_over = env.frame_step(best_action)
 			steps += 1
 			total_reward += reward
 			print reward
 			if ploc == self.env.goalloc:
 				break
 		return total_reward, steps
