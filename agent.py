import torch
import random
import numpy as np
from model import QNetwork
import random
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Experiment hyperparameters
LR = 0.001
EPS = 0.1
MIN_EPS = 0.01
EPS_DECAY = 0.99
ALPHA = 0.2
GAMMA = 0.99

BUFFER_SIZE = 10000
BATCH_SIZE = 100


class DqnAgent():
    '''Agent that will interact with our environnement'''

    def __init__(self, state_size, action_size, seed=None):
        if seed:
            self.seed = random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.eps = EPS

        # Memmory to learn from.
        self.memory = ReplayBuffer(memory_size=BUFFER_SIZE, sample_size=BATCH_SIZE)

        # Network
        self.qnetwork = QNetwork(state_size=state_size, action_size=action_size).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=LR)

    def learn(self, transition_infos):
        '''
            Update the policy of the agent from samples memory.
            Input:
                transition_info(tuple): list of Tuple of the format (s, a, r, s')
        '''
        state, action, reward, next_state, done = transition_infos

        # Get old estimate of the q_values according to doing old action while being in the old state.
        Qva = self.qnetwork(state).data[action]

        # Get new estimated q values in states s' while still following policy.
        max_espected_value = torch.max(self.qnetwork(next_state))

        # if done, then there is no next Q value estimate to make.
        max_espected_value = (1 - done) * max_espected_value

        # reseting optimizer at each update step
        self.optimizer.zero_grad()

        update_loss = Qva + ALPHA * (reward + GAMMA * max_espected_value - Qva)

        update_loss.backward()

        # update
        self.optimizer.step()

        return torch.mean(update_loss).data.cpu().numpy()

    def act(self, state, episode):
        '''
            Select an action based on inference from the state following an e-greedy policy.
            Input:
                state(np.Array): The representation of the state from the environnement.
            Return:
                action(int): Action
        '''
        # decay epsilon
        self.eps = self.eps * EPS_DECAY

        actions = self.qnetwork.forward(state)

        if random.random() > max(MIN_EPS, EPS):
            # Select highest Q values action
            action = np.argmax(np.argmax(actions.cpu().detach().numpy()))
        else:
            # Select a random actions
            action = random.choice(np.arange(self.action_size))

        return action

class ReplayBuffer():
    def __init__(self, memory_size, sample_size):
        '''Stores tuples of (state, action, reward, next_state, done)'''
        # self.memory = namedtuple('Memory', ['state', 'action', 'reward', 'next_state', 'done'])
        self.experience = deque(maxlen=memory_size)
        
        self.sample_size = sample_size

    def sample(self):
        '''Return (sample_size) elements from memory.'''
        sampling = random.choices(self.experience, k=self.sample_size)

        # transposing row of tuples to tuples of rows
        sampling = np.transpose(np.array(sampling))

        states = sampling[0, :]
        actions = sampling[1, :]
        rewards = sampling[2, :]
        next_states = sampling[3, :]
        dones = sampling[4, :]

        return (states, actions, rewards, next_states, dones)

    def add_recollection(self, memory):
        self.experience.append(memory)
