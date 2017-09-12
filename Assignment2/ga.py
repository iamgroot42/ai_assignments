import numpy as np
from tqdm import tqdm

class GeneticAlgorithm:
	def __init__(self, weights, pop_size, num_iters, selection_ratio):
		self.pop_size = pop_size
		self.num_iters = num_iters
		self.selection_ratio = selection_ratio
		self.weights = weights

	def permutation_weight(self, permutation):
		cost = [ self.weights[i][permutation[i]] for i in range(self.weights.shape[0])]
		return sum(cost)

	def siblings(self, permA, permB, mementic=None):
		child1 = list(permA[:len(permA)/2])
		child2 = list(permB[:len(permB)/2])
		parentA = list(set(list(permA)) - set(list(child2)))
		parentB = list(set(list(permB)) - set(list(child1)))
		for i in range(len(permB)-len(child1)):
			child1.append(parentB[i])
		for i in range(len(permA)-len(child2)):
			child2.append(parentA[i])
		child1 = np.array(child1)
		child2 = np.array(child2)
		if  mementic:
			costs = []
			indices = []
			for _ in range(50):
				indices.append(np.random.choice(len(child1),len(child1),replace=False))
				costs.append(self.permutation_weight(child1[indices[-1]]))
			top_2 = np.argsort(costs)[::-1][:2]
			child1 = child1[indices[top_2[0]]]
			child2 = child2[indices[top_2[1]]]
		else:
			np.random.shuffle(child1)
                        np.random.shuffle(child2)	
		return child1, child2


	def combine(self, permutations):
		new_permutations = []
		parents = np.random.choice(len(permutations), len(permutations), replace=False)
		mothers = np.take(permutations, parents[:len(permutations)/2], axis=0)
		fathers = np.take(permutations, parents[len(permutations)/2:], axis=0)
		for i in range(len(mothers)):
			sister, brother = self.siblings(mothers[i], fathers[i], True)
			new_permutations.append(sister)
			new_permutations.append(brother)
		return np.concatenate((permutations, np.array(new_permutations)))


	def evolve(self):
		permutations = []
		# Create initial population
		temp_permutations = []
		for _ in range(self.pop_size):
			permutations.append(np.random.choice(self.weights.shape[1], self.weights.shape[0], replace=False))
		# Across iterations of evolution
		for _ in tqdm(range(self.num_iters)):
			# Calculate costs for all permutations
			permutation_weights = np.array([self.permutation_weight(x) for x in permutations])
			# Pick the top selection_ratio permutations for repopulation
			permutations = np.take(permutations, np.argsort(permutation_weights)[::-1][:int(self.selection_ratio * len(permutations))], axis=0)
			# Repopulate and add new permutations
			permutations = self.combine(permutations)
		
		# Pick the best permutation and return it
		permutation_weights = np.array([self.permutation_weight(x) for x in permutations])
		optimal_permutation = np.take(permutations, np.argsort(permutation_weights), axis=0)[-1]
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
	ga = GeneticAlgorithm(values, 2000, 2000, 0.5)
	print ga.evolve()
