import numpy as np
from tqdm import tqdm

from ga import GeneticAlgorithm

class AntColony:
	def __init__(self, weights, n_ants, constant=10, alpha=1, beta=5, Q=100, rho=0.99):
		self.weights = weights
		self.alpha = alpha
		self.beta = beta
		self.n_ants = n_ants
		self.rho = rho
		self.tau = np.ones(self.weights.shape) * constant
		self.tau_delta = np.zeros(self.weights.shape)
		self.tabu = []
		self.shortest_tour = []
		self.shortest_tour_cost = np.sum(self.weights)
		self.Q = Q

	def empty_tabu(self):
		self.tabu = []
		for _ in range(self.n_ants):
			self.tabu.append([])
		for i in range(self.n_ants):
			randnum = np.random.randint(0, weights.shape[0])
			self.tabu[i].append(randnum)

	def populate_tabu(self):
		for s in range(1,self.weights.shape[0]):
			for k in range(self.n_ants):
				i = self.tabu[k][-1]
				p_ij_values = []
				valid_dests = []
				for j in range(self.weights.shape[0]):
					if j not in self.tabu[k]:
						valid_dests.append(j)
						p_ij_values.append(np.power(self.tau[i][j], self.alpha)*np.power(1.0/self.weights[i][j],self.beta))
				denominator = sum(p_ij_values)
				for j in range(len(p_ij_values)):
					p_ij_values[j] /= denominator
				picked_city = valid_dests[np.random.choice(len(valid_dests),replace=False,p=p_ij_values)]
				self.tabu[k].append(picked_city)

	def get_tour_cost(self, tour):
		assert(len(tour) == self.weights.shape[0])
		cost = 0
		for i in range(1,len(tour)):
			cost += self.weights[tour[i]][tour[i-1]]
		cost += self.weights[tour[-1]][tour[0]]
		return cost

	def edge_in_tour(self, tour, x, y):
		is_part = False
		for i in range(1,len(tour)):
			if (tour[i] == x and tour[i-1] == y):
				is_part = True
		if (tour[-1] == x and tour[0] == y):
			is_part = True
		return False

	def ant_move(self):
		L_k = []
		for k in range(self.n_ants):
			L_k.append(self.get_tour_cost(self.tabu[k]))
			if L_k[-1] < self.shortest_tour_cost:
				self.shortest_tour = self.tabu[k]
				self.shortest_tour_cost = L_k[-1]
		for i in range(self.weights.shape[0]):
			for j in range(self.weights.shape[0]):
				for k in range(self.n_ants):
					if self.edge_in_tour(self.tabu[k], i, j):
						self.delta_tau[i][j] += self.Q/L_k[k]

	def evaporate(self):
		for i in range(self.weights.shape[0]):
			for j in range(self.weights.shape[0]):
				self.tau[i][j] = self.rho * self.tau[i][j] + self.tau_delta[i][j]
				self.tau_delta[i][j] = 0

	def run_simulation(self, nc_max):
		for i in tqdm(range(nc_max)):
			self.empty_tabu()
			self.populate_tabu()
			self.ant_move()
			self.evaporate()
		return self.shortest_tour_cost, self.shortest_tour


if __name__ == "__main__":
	weights = np.random.randint(1,100,(20,20))
	ac = AntColony(weights, 100)
	print ac.run_simulation(200)
	ga = GeneticAlgorithm(weights, 200, 200, 0.5, True, True)
	print ga.evolve()
