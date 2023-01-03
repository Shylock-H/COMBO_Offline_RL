import gym
import d4rl

env = gym.make('hopper-medium-replay-v2')
dataset = d4rl.qlearning_dataset(env)
print(d4rl.get_normalized_score('halfcheetah-medium-replay-v2', 5000) * 100)