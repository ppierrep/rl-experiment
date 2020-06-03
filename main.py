from agent import DqnAgent
from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="env/Banana_Linux_NoVis/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of actions
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
update_at = 10

agent = DqnAgent(state_size=len(state), action_size=action_size)

scores = []
loses = []

# Instanciate a new agent.
agent = DqnAgent(state_size=len(state), action_size=action_size)

for episode in range(500):
    done = False
    score = 0
    loss = 0

    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]

    while True:
        action = agent.act(state)                              # ask agent action selection
        env_info = env.step(action)[brain_name]                # send the action to the environment

        next_state = env_info.vector_observations[0]           # get the next state
        reward = env_info.rewards[0]                           # get the reward
        done = env_info.local_done[0]                          # see if episode has finished
        score += reward                                        # update the score
        transition_info = (state, action, reward, next_state)

        loss += agent.learn(transition_info, done)             # learn from transition
        score += reward

        if done:
            scores.append(score)
            loses.append(loss)
            break

        state = next_state                                     # roll over the state to next time step

    if episode % update_at == 0:
        plot(scores, loses)
        print('Episode: {} - '.format(episode), end="")
        print('Averaged Score of the last {} episodes : {}'.format(update_at, np.mean(scores[-update_at:]).round(2)))

env.close()