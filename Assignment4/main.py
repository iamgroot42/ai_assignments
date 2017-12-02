import matplotlib.pyplot as plt

import grid
import valueIteration
import Qlearning


if __name__ == "__main__":
	# First grid:
	env = grid.gridenv()
	# Second grid:
	env2 = grid.gridenv()
	# Combined grid
	env3 = grid.gridenv()
	# Probabilistic grid
	env4 = grid.gridenv()

	# Q-learning on first grid, test on first
	ploc, x = env.initDeterministicgrid(1)
	# v = valueIteration.ValueIteration(env, x, 0.5)
	# v.runIterations()
	# print v.run_agent(ploc, env)
	q = Qlearning.Qlearning(env, x)
	rewards = q.runIterations()
	print "?"
	env2.initProbalisticgrid(1)
	print "done"
	print q.run_agent(ploc, env2)



	exit()

	# Plot for first part
	plt.plot(rewards)
	plt.ylabel('Reward')
	plt.xlabel('Episode')
	plt.title('Reward per episode for Q learning')
	plt.show()

	# Q-learning on first grid, test on second
	ploc, x = env.initDeterministicgrid(1)
	ploc_, x_ = env2.initDeterministicgrid(2)
	q = Qlearning.Qlearning(env, x)
	rewards = q.runIterations()
	q.run_agent(ploc_, env2)

	# Q-learning on both grids combination, test on second
	ploc, x = env3.initDeterministicgrid(3)
	ploc_, x_ = env.initDeterministicgrid(2)
	q = Qlearning.Qlearning(env3, x)
	rewards = q.runIterations()
	q.run_agent(ploc_, env2)

	# Value iteration on both grids, test on second
	ploc, x = env3.initDeterministicgrid(3)
	ploc_, x_ = env2.initDeterministicgrid(2)
	v = valueIteration.ValueIteration(env3, x, 0.5)
	v.runIterations()
	v.run_agent(ploc_, env2)


	# Value iteration on probabilistic
	ploc, x = env3.initDeterministicgrid(3)
	ploc_, x_ = env4.initProbalisticgrid()
	v = valueIteration.ValueIteration(env3, x, 0.5)
	v.runIterations()
	v.run_agent(ploc_, env4)

	# Q-learning on probabilistic
	ploc, x = env3.initDeterministicgrid(3)
	ploc_, x_ = env4.initProbalisticgrid()
	q = Qlearning.Qlearning(env3, x)
	rewards = q.runIterations()
	q.run_agent(ploc_, env4)
