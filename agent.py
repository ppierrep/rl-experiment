import torch
import random
import numpy as np
from model import QNetwork


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Experiment hyperparameters
LR = 0.001
EPS = 0.1
ALPHA = 0.2
GAMMA = 0.99


class DqnAgent():
    '''Agent that will interact with our environnement'''

    def __init__(self, state_size, action_size, seed=None):
        if seed:
            self.seed = random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        # Network
        self.qnetwork = QNetwork(state_size=state_size, action_size=action_size).to(device)
        self.optimizer = torch.optim.SGD(self.qnetwork.parameters(), lr=LR)

    def learn(self, transition_info, done):
        '''
            Update the policy of the agent from one step infos.
            Input:
                transition_info(tuple): Tuple of the format (s, a, r, s')
        '''
        state, action, reward, next_state = transition_info

        # Get old estimate of the q_values according to doing old action while being in the old state.
        Qva = self.qnetwork(state)

        # Get new estimated q values in states s' while still following policy.
        max_espected_value = torch.max(self.qnetwork(next_state).detach())

        # if done, then there is no next Q value estimate to make.
        max_espected_value = (1 - done) * max_espected_value

        update_loss = Qva + ALPHA * (reward + GAMMA * max_espected_value - Qva)

        # reseting optimizer at each update step
        self.optimizer.zero_grad()
        Qva.backward(update_loss.data.unsqueeze(1))

        # update
        self.optimizer.step()

    def act(self, state):
        '''
            Select an action based on inference from the state following an e-greedy policy.
            Input:
                state(np.Array): The representation of the state from the environnement.
            Return:
                action(int): Action
        '''
        actions = self.qnetwork.forward(state)

        if random.random() > EPS:
            # Select highest Q values action
            action = np.argmax(np.argmax(actions.detach().numpy()))
        else:
            # Select a random actions
            action = random.choice(np.arange(self.action_size))

        return action
