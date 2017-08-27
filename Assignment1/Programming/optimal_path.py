import numpy as np


class Grid:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.grid = np.zeros((x, y))

	def get_neighbors(self, i, j):
		neighbors = []
		for a in range(-1,2):
			for b in range(-1,2):
				if i+a>=0 and i+a<self.x and j+b>=0 and j+b<self.y:
					# if a*b ==0:
						neighbors.append((i+a, b+j))
		return neighbors

	def populate_restricted_cells(self, regions):
		for region in regions:
			x, y, w, h = region
			for i in range(w):
				for j in range(h):
					self.grid[x+i][y+j] = -1

	def search(self, start_x=0, start_y=0, DFS=False):
		distances = np.zeros((self.x, self.y))
		visited = np.zeros((self.x, self.y))
		visited[start_x][start_y] = 1
		q = []
		q.append((start_x,start_y))
		while q:
			if DFS:
				i,j = q.pop()
			else:	
				i,j = q.pop(0)
			for neighbor in self.get_neighbors(i,j):
				a,b = neighbor
				if visited[a][b] == 0 and self.grid[a][b]!= -1:
					q.append((a,b))
					distances[a][b] = distances[i][j] + 1
					visited[a][b] = 1
		return distances


def parse_data(filename, rectangles=False):
	coordinates = []
	with open(filename, 'r') as f:
		for data in f:
			if rectangles:
				x, y, w, h  = data.rstrip().split(' ')
				x, y, w, h = int(round(float(x))), int(round(float(y))), int(round(float(w))), int(round(float(h)))
				coordinates.append((x, y, w, h))
			else:
				x, y  = data.rstrip().split(' ')
				x, y  = int(round(float(x))), int(round(float(y)))
				coordinates.append((x, y))
	return coordinates


def pick_employees(g, points, DFS=False):
	employees = [0 for i in range(len(points))]
	order = []
	x,y = 0,0
	cost = 0
	while sum(employees) != len(employees):
		bfs = g.search(x,y, DFS)
		distances = [ bfs[i][j] for (i,j) in points]
		indices = np.argsort(distances)
		for i in indices:
			if employees[i] == 0:
				min_index = i
				break
		x,y = points[min_index]
		order.append(min_index)
		cost += distances[min_index]
		employees[min_index] = 1
	cost += g.search(x,y,DFS)[g.x-1][g.y-1]
	return cost, order


if __name__ == "__main__":
	import sys
	DFS=False
	data = parse_data(sys.argv[1])
	graph = Grid(50, 50)
	regions = parse_data(sys.argv[2], True)
	if sys.argv[3].lower() == 'dfs':
		DFS=True
	graph.populate_restricted_cells(regions)
	cost, order = pick_employees(graph, data, DFS)
	print "Approximaetd total cost of current search:",cost
	print "Order to be visited in:", order
