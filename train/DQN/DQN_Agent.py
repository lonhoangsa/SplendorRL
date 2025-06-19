import torch
import torch.optim as optim
from collections import deque
import random
import numpy as np
import torch.nn as nn
from train.DQN.DQN import DQN

class DQNAgent:
    def __init__(self, state_dim, action_dim, device='cpu', logger=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.logger = logger

        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)
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
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_mask_tensor = torch.FloatTensor(action_mask).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values = q_values.masked_fill(action_mask_tensor == 0, float('-inf'))
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        self.memory.append((state, action, reward, next_state, done, action_mask, next_action_mask))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = zip(*batch)
        
        # Chuyển list thành mảng NumPy với kích thước đồng nhất
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        action_masks = torch.FloatTensor(np.stack(action_masks)).to(self.device)
        next_action_masks = torch.FloatTensor(np.stack(next_action_masks)).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.masked_fill(next_action_masks == 0, float('-inf'))
            next_q_values = next_q_values.max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.logger.info(f"Target network updated at step {self.steps}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def get_best_action(self, state, action_mask=None):
        """
        Get the best action based on Q-value predictions
        Args:
            state: The game state
            action_mask: Optional mask for legal actions
        Returns:
            best_action: The action with highest Q-value
            q_values: Q-values for all actions
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_mask_tensor = torch.FloatTensor(action_mask).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values = q_values.masked_fill(action_mask_tensor == 0, float('-inf'))
            return q_values.argmax().item()