import numpy as np
from unityagents import UnityEnvironment
from utils import plot

import torch

from agent import DqnAgent


env = UnityEnvironment(file_name="env/Banana_Linux_NoVis/Banana.x86_64")

dqn_params = {
    # Experiment hyperparameters
    'LR': 5e-4,               # learning rate
    
    'EPS': 1,                 # epsilon greedy start value
    'MIN_EPS': 0.01,          # epsilon greedy minimal
    'EPS_DECAY': 0.995,

    'ALPHA': 0.2,             # Alpha TD-error
    'GAMMA': 0.98,            # Gamma Td-error
    'TAU': 1e-3,              # for soft update of target parameters

    'UPDATE_EVERY': 4,        # Update fixed target model 

    'BUFFER_SIZE': int(1e4),  # Memmory max size
    'BATCH_SIZE': 64,         # Number of episode from which fixed target model learn from at each step
}

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

update_every = 100
max_time_per_episode = 1000

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]

agent = DqnAgent(state_size=len(state), action_size=action_size, params=dqn_params)

scores = []
loses = []
action_takens = []

for episode in range(3000):
    score = 0
    loss = 0
    action_taken = []

    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]

    for max_t in range(max_time_per_episode):
        action = agent.act(state)                     # ask agent action selection
        env_info = env.step(action)[brain_name]                # send the action to the environment

        next_state = env_info.vector_observations[0]           # get the next state
        reward = env_info.rewards[0]                           # get the reward
        done = env_info.local_done[0]                          # see if episode has finished

        transition_info = (state, action, reward, next_state, done)
        loss += agent.step(transition_info)

        score += reward
        action_taken.append(action)

        if done:
            scores.append(score)
            loses.append(loss)
            action_takens += action_taken
            break

        state = next_state

    if episode % update_every == 0:
        plot(scores, loses, action_takens)
        print('Episode: {} - '.format(episode), end="")
        print('Averaged Score of the last {} episodes : {}'.format(update_every, np.mean(scores[-update_every:]).round(2)))
        torch.save(agent.local.state_dict(), 'checkpoint.pth')

env.close()