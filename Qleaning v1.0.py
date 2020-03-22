import gym
import numpy
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.reset()
done = False

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

Q_table_size = [100] * len(env.observation_space.high)  # bigger the better, might need to lower depends on ram usage
range_size = (env.observation_space.high - env.observation_space.low) / Q_table_size

print(range_size)

q_table = numpy.random.uniform(Low=-2, high=0, size=(Q_table_size + [env.action_space.n]))

'''
while not done:
    action = 1
    new_state, reward, done, hold = env.step(action)
    print(reward, new_state)
    env.render()

env.close()

'''
