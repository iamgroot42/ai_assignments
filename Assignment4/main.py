import grid
import valueIteration


if __name__ == "__main__":
	env = grid.gridenv()
	x = env.initDeterministicgrid()
	state, reward, game_over = env.frame_step(3) # input is the action number
	v = valueIteration.ValueIteration(env, x)
	v.runIterations()
