import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
import random
import numpy as np
from train.DuelingDQN import DuelingDQN

class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, device='cpu', logger=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.logger = logger

        # Use the Dueling DQN network
        self.q_network = DuelingDQN(state_dim, action_dim).to(device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=100000)
        self.batch_size = 512
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        self.steps = 0

    def act(self, state, action_mask):
        if np.random.random() < self.epsilon:
            legal_actions = np.where(action_mask)[0]
            return np.random.choice(legal_actions)
        
        # Add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            # Mask illegal actions
            q_values = q_values.masked_fill(action_mask_tensor == 0, float('-inf'))
            return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        self.memory.append((state, action, reward, next_state, done, action_mask, next_action_mask))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = zip(*batch)
        
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        action_masks = torch.FloatTensor(np.stack(action_masks)).to(self.device)
        next_action_masks = torch.FloatTensor(np.stack(next_action_masks)).to(self.device)

        # Current Q-value for selected actions
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: choose next action with the online network; evaluate with target network
        with torch.no_grad():
            # Get Q-values from the online network and mask illegal actions
            next_q_values_online = self.q_network(next_states)
            next_q_values_online = next_q_values_online.masked_fill(next_action_masks == 0, float('-inf'))
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

            # Use target network to evaluate the chosen actions
            next_q_values_target = self.target_network(next_states)
            next_q_values_target = next_q_values_target.masked_fill(next_action_masks == 0, float('-inf'))
            next_q_value = next_q_values_target.gather(1, next_actions).squeeze(1)

            targets = rewards + self.gamma * next_q_value * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            if self.logger is not None:
                self.logger.info(f"Target network updated at step {self.steps}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()