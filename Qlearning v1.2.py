import gym
import numpy
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')  # picks the environment


learning_rate = 0.1
how_important_future_actions = 0.95
times_to_run = 25000     # increase if q table size is better

show_every = 2000

Q_table_size = [20] * len(env.observation_space.high)  # number of Columns     bigger the better, might need to lower depends on ram usage    if this is higher you need higher time to run
range_size = (env.observation_space.high - env.observation_space.low) / Q_table_size  # how far to carry the decimal point out too

epsilon = .5  # randomness
start_epsilon_decaying = 1
end_epislon_decaying = times_to_run // 2
epsilon_decay_value = epsilon/(end_epislon_decaying - start_epsilon_decaying)  # calacuted it

q_table = numpy.random.uniform(low=-2, high=0, size=(Q_table_size + [env.action_space.n])) # makes the table


def get_state(state):
    the_state = (state - env.observation_space.low) / range_size  # gets a state for the qtable
    return tuple(the_state.astype(numpy.int))

for run in range(times_to_run):

    if run % show_every == 0:
        render = True
        print(run)
    else:
        render = False

    the_state = get_state(env.reset()) # gets a state for the qtable
    done = False
    while not done:

        if numpy.random.random() > epsilon:
            action = numpy.argmax(q_table[the_state])  # gets action from qtable by getting max reward action
        else:
            action = numpy.random.randint(0, env.action_space.n)
        new_state, reward, done, hold = env.step(action)  # updates
        new_the_state = get_state(new_state)  # the new state for forumla
        if render:
            env.render()
        if not done:
            max_furture = numpy.max(q_table[new_the_state])  # for forumla
            current = q_table[the_state + (action, )]

            new_q = (1 - learning_rate) * current + learning_rate * (reward + how_important_future_actions * max_furture)  # formula
            q_table[the_state+(action, )] = new_q  # updating qtable
        elif new_state[0] >= env.goal_position:
            print(f"We made it on turn {run}")
            q_table[the_state + (action, )] = 0 # reward for winning

        the_state = new_the_state

    if end_epislon_decaying >= run >= start_epsilon_decaying: # decayed it
        epsilon -= epsilon_decay_value

env.close()



'''
helpful stuff

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

print(range_size)


numpy.argmax(q_table[the_state]) # picks the best action
print(numpy.argmax(q_table[the_state]))
'''
