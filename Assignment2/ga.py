import numpy as np
from tqdm import tqdm


class GeneticAlgorithm:
	def __init__(self, weights, pop_size, num_iters, selection_ratio, random_ratio):
		self.pop_size = pop_size
		self.num_iters = num_iters
		self.selection_ratio = selection_ratio
		self.random_ratio = random_ratio
		self.weights = weights

	def permutation_weight(self, permutation):
		cost = [ self.weights[i][permutation[i]] for i in range(len(permutation))]
		return sum(cost)

	def combine(self, permutations):
		new_permutations = []
		parents = np.random.choice(len(permutations), len(permutations), replace=False)
		mothers = np.take(permutations, parents[:len(permutations)/2], axis=0)
		fathers = np.take(permutations, parents[len(permutations)/2:], axis=0)
		for i in range(len(mothers)):
			selected_genes = np.random.choice(len(mothers[i]), int(len(mothers[i])*self.random_ratio),replace=False)
			temp_perm = np.array(fathers[i])
			temp_perm[selected_genes] = mothers[i][selected_genes]
			new_permutations.append(temp_perm)
		return np.concatenate((permutations, np.array(new_permutations)))


	def evolve(self):
		permutations = []
		# Create initial population
		temp_permutations = []
		for _ in range(self.pop_size):
			permutations.append(np.random.choice(self.weights.shape[1], self.weights.shape[0]))
		# Across iterations of evolution
		for _ in tqdm(range(self.num_iters)):
			# Calculate costs for all permutations
			permutation_weights = np.array([self.permutation_weight(x) for x in permutations])
			# Pick the top selection_ratio permutations for repopulation
			permutations = np.take(permutations, np.argsort(permutation_weights)[:int(self.selection_ratio * len(permutations))], axis=0)
			# Repopulate and add new permutations
			permutations = self.combine(permutations)
		# Pick the best permutation and return it
		permutation_weights = np.array([self.permutation_weight(x) for x in permutations])
		optimal_permutation = np.take(permutations, np.argsort(permutation_weights)[::-1], axis=0)[0]
		return permutation_weights.max(), optimal_permutation


def process_data(filepath):
	processes = []
	people = []
	values = []
	with open(filepath, 'r') as f:
		processes = f.readline().rstrip().split(' ')
		for line in f:
			data = line.rstrip().split(' ')
			people.append(data[0])
			values.append([float(x) for x in data[1:]])
	return processes, people, np.array(values)


if __name__ == "__main__":
	import sys
	processes, people, values = process_data(sys.argv[1])
	values = np.transpose(values)
	ga = GeneticAlgorithm(values, 30, 30, 0.95, 0.5)
	print ga.evolve()
