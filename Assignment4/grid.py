import numpy as np
from collections import deque

player = 1
goal = 2
wall = -1
safe = 0

class gridenv(object):
	
	def __init__(self):
		self.GRID_SIZE = 5
		self.game_over = False
		self.grid = np.zeros((self.GRID_SIZE,self.GRID_SIZE))
		self.frames = np.zeros((4,5,5))
		
	def resetEnv(self):
		self.grid = np.zeros((self.GRID_SIZE,self.GRID_SIZE))
		self.frames = np.zeros((4,5,5))

	def defineHeuristic(self):
		self.hueuristicgrid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
		for i in range(self.GRID_SIZE):
			for j in range(self.GRID_SIZE):
				self.hueuristicgrid[i, j] = 2 * (self.GRID_SIZE - 1) - (i + j)

	def get_reward(self, location):
		print location
		reward = 0
		location = self.grid[location[1], location[0]]
		if location == goal:
			reward = 300
		elif location == safe:
			reward=0
			# reward = -1*(self.hueuristicgrid[self.ploc[1],self.ploc[0]])
		elif location == wall:
			reward = -100
		else:
			# reward = -1*(self.hueuristicgrid[self.ploc[1],self.ploc[0]])
			reward=0
		return reward

	def returnFrames(self):
		return np.array(self.frames)

	def reset_state(self, ploc, grid):
		self.ploc = ploc
		self.grid = grid

	def initDeterministicgrid(self, other_grid=1):
		self.defineHeuristic()
	
		self.no_walls = 3

		if other_grid == 1:
			self.walls = [(1,1),(2,2),(1,3)]
		elif other_grid == 1:
			self.walls = [(1,2),(4,1),(3,3)]
		else:
			self.walls = [(1,1),(1,2),(2,2),(1,3),(3,3),(4,1)]
		self.ploc = [0,0]
		self.goalloc = [self.GRID_SIZE-1,self.GRID_SIZE-1]
		
		for wall_ in self.walls:
			self.grid[wall_[1],wall_[0]] = wall
		
		self.grid[self.goalloc[1],self.goalloc[0]] = goal

		px, py = 0, 0
		self.grid[py,px] = player
		self.ploc = [px,py]
		
		for x in range(4):
			self.frames[x,] = self.grid
			
		return self.ploc, self.grid
		
	def initProbalisticgrid(self,prob=0.1):
		
		self.resetEnv()
		self.defineHeuristic()
		self.goalloc = [self.GRID_SIZE-1,self.GRID_SIZE-1]
		self.grid[self.goalloc[1],self.goalloc[0]] = goal
		
		self.probability = prob
		for x in range(self.GRID_SIZE):
			for y in range(self.GRID_SIZE):
				if (self.grid[y][x]==0):
					if np.random.rand() < self.probability:
						self.grid[y,x] = wall
		tmp_p = False
		while(tmp_p!=True):
			print "try"
			px = np.random.randint(low=0,high=self.GRID_SIZE,size=1)
			py = np.random.randint(low=0,high=self.GRID_SIZE,size=1)
			if (px == self.GRID_SIZE-1 and py == self.GRID_SIZE-1) or (self.grid[py,px] == wall):
				px = np.random.randint(low=0,high=self.GRID_SIZE,size=1)
				py = np.random.randint(low=0,high=self.GRID_SIZE,size=1)
			else:
				tmp_p = True
		self.grid[py,px] = player
		self.ploc = [px,py]

		for x in range(4):
			self.frames[x,] = np.copy(self.grid)
		
		return self.ploc, self.frames
	
	def get_action_state(self, state, action):

		if(action == 0): # 0 - move backward
			if(state[1]!=0):
				return (state[1]-1,state[0])
		elif(action == 1): # 1 - move forward
			if(state[1]!=self.GRID_SIZE-1):
				return (state[1]+1,state[0])
		elif(action == 2): # 2 - move left
			if(state[0]!=0):
				return (state[1],state[0]-1)
		elif(action == 3): # 3 - move right
			if(state[0]!=self.GRID_SIZE-1):
				return (state[1],state[0]+1)
		return (state[1],state[0])


	def frame_step(self,action):
		
		self.game_over = False
		self.reward = 0
		old_ploc = self.ploc
		self.grid[old_ploc[1],old_ploc[0]] = safe  # removed player from grid
		prevValue = self.ploc  # it signifies the cell value of new player location (here location though)
		tmp = None
		notAllowedLoc = 0
		
		if(action == 0): # 0 - move backward
			if(self.ploc[1]!=0):
				self.ploc[1] -= 1
				prevValue = self.grid[self.ploc[1],self.ploc[0]]
				tmp = 0
			else: 
				notAllowedLoc = 1
				
		elif(action == 1): # 1 - move forward
			if(self.ploc[1]!=self.GRID_SIZE-1):
				self.ploc[1] += 1
				prevValue = self.grid[self.ploc[1],self.ploc[0]]
				tmp = 1
			else: 
				notAllowedLoc = 1
				
		elif(action == 2): # 2 - move left
			if(self.ploc[0]!=0):
				self.ploc[0] -= 1
				prevValue = self.grid[self.ploc[1],self.ploc[0]]
				tmp = 2
			else: 
				notAllowedLoc = 1
				
		elif(action == 3): # 3 - move right
			if(self.ploc[0]!=self.GRID_SIZE-1):
				self.ploc[0] += 1
				prevValue = self.grid[self.ploc[1],self.ploc[0]]
				tmp = 3
			else: 
				notAllowedLoc = 1

		if prevValue == goal:  # if new player cell is goal
			self.grid[self.ploc[1],self.ploc[0]] = 3
			self.reward = 300
			self.game_over = True
		elif prevValue == safe and notAllowedLoc == 0:
			self.grid[self.ploc[1],self.ploc[0]] = player
			self.reward = -1*(self.hueuristicgrid[self.ploc[1],self.ploc[0]])
			self.game_over = False
		elif prevValue == wall:
			self.grid[self.ploc[1],self.ploc[0]] = wall
			self.reward = -100
			self.game_over = False
			# reset player to same location
			if tmp == 0:
				self.grid[self.ploc[1]+1,self.ploc[0]] = player
				self.ploc[1] += 1
			elif tmp == 1:
				self.grid[self.ploc[1]-1,self.ploc[0]] = player
				self.ploc[1] -= 1
			elif tmp == 2:
				self.grid[self.ploc[1],self.ploc[0]+1] = player
				self.ploc[0] += 1
			elif tmp == 3:
				self.grid[self.ploc[1],self.ploc[0]-1] = player
				self.ploc[0] -= 1
			
		else:
			self.grid[self.ploc[1],self.ploc[0]] = player
			self.reward = -1*(self.hueuristicgrid[self.ploc[1],self.ploc[0]])
			
		if notAllowedLoc == 1:
			self.grid[self.ploc[1],self.ploc[0]] = player
			self.reward = -100
			self.game_over = False

		self.frames[0:3,] = np.copy(self.frames[1:4,])
		self.frames[3,] = np.copy(self.grid)
		return self.ploc, self.grid, self.reward, self.game_over
