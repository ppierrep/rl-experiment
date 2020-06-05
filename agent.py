import torch
import random
import numpy as np
from model import QNetwork
import random
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DqnAgent():
    '''Agent that will interact with our environnement'''

    def __init__(self, state_size, action_size, params, seed=None):
        self.seed = seed
        if seed:
            random.seed(seed)
            np.random.seed(0)

        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.eps = self.params['EPS']
        
        # Memmory to learn from.
        self.memory = ReplayBuffer(memory_size=self.params['BUFFER_SIZE'], sample_size=self.params['BATCH_SIZE'])

        # Network
        self.target = QNetwork(state_size=state_size, action_size=action_size, seed=seed).to(device)
        self.local = QNetwork(state_size=state_size, action_size=action_size, seed=seed).to(device)
        self.optimizer = torch.optim.Adam(self.local.parameters(), lr=self.params['LR'])
       
        self.t_step = 0

    def learn(self, transition_info):
        '''
            Update the policy of the agent from one step transition.
            Input:
                transition_info(tuple): Tuple of the format (s, a, r, s')
        '''
        state, action, reward, next_state, done = transition_info

        # Get old estimate of the q_values according to doing old action while being in the old state.
        Q_target = self.target(next_state).detach().max(1)[0].unsqueeze(1)
        Q_target = reward.unsqueeze(1) + self.params['GAMMA'] * Q_target * (1 - done.unsqueeze(1))

        # Get expected Q values from local model
        Q_expected = self.local(state).gather(1, action.unsqueeze(1))

        # Compute loss
        loss = torch.nn.functional.mse_loss(Q_expected, Q_target)

        # reseting optimizer at each update step
        self.optimizer.zero_grad()
        loss.backward()

        # update
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.local, self.target, self.params['TAU'])
        
        # Return the loss for monitoring
        return torch.mean(loss).data.cpu().numpy()

    def act(self, state):
        '''
            Select an action based on inference from the state following an e-greedy policy.
            Input:
                state(np.Array): The representation of the state from the environnement.
            Return:
                action(int): Action
        '''
        # decay epsilon
        self.eps = self.eps * self.params['EPS_DECAY']
        e = max(self.params['MIN_EPS'], self.eps)

        state = torch.from_numpy(state).detach().to(device).unsqueeze(0).float()
        
        self.local.train(mode=False)
        with torch.no_grad():
            actions = self.local(state)
        self.local.train(mode=True)
            
        if random.random() > e:
            # Select highest Q values action
            action = np.argmax(actions.cpu().detach().numpy())
        else:
            # Select a random actions
            action = random.choice(np.arange(self.action_size))

        return action

    def step(self, transition_info):
        loss = 0
        # Save experience in replay memory
        self.memory.add_recollection(transition_info)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.params['UPDATE_EVERY']
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.params['BATCH_SIZE']:
                samples = self.memory.sample()
                loss += self.learn(samples)
        return loss

    def soft_update(self, local_model, target_model, tau):
            """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            Params
            ======
                local_model (PyTorch model): weights will be copied from
                target_model (PyTorch model): weights will be copied to
                tau (float): interpolation parameter 
            """
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
class ReplayBuffer():
    def __init__(self, memory_size, sample_size):
        '''Stores tuples of (state, action, reward, next_state, done)'''
        self.experience = deque(maxlen=memory_size)
        self.sample_size = sample_size

    def sample(self):
        '''Return (sample_size) elements from memory.'''
        
        sampling = random.choices(self.experience, k=self.sample_size)

        # transposing row of tuples to tuples of rows
        sampling = np.transpose(np.array(sampling))

        states = torch.from_numpy(np.stack(sampling[0, :])).float().to(device)
        actions = torch.from_numpy(np.stack(sampling[1, :])).long().to(device)
        rewards = torch.from_numpy(np.stack(sampling[2, :])).float().to(device)
        next_states = torch.from_numpy(np.stack(sampling[3, :])).float().to(device)
        dones = torch.from_numpy(np.stack(sampling[4, :]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def add_recollection(self, memory):
        self.experience.append(memory)
        
    def __len__(self):
        return len(self.experience)