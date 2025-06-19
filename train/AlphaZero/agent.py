import torch
import numpy as np
from train.AlphaZero.network import AlphaZeroNet
from train.AlphaZero.mcts import MCTS
import logging
from env.splendor_lightzero_env import SplendorLightZeroEnv

class AlphaZeroAgent:
    def __init__(self, state_dim, action_dim, config=None, device=None, architecture_type='simple'):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaZeroNet(state_dim, action_dim, architecture_type=architecture_type, device=self.device)
        self.mcts = MCTS(self.net, config)
        self.config = config
        self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                        lr=config.get('lr', 0.001),
                                        weight_decay=config.get('weight_decay', 0.0001))
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized AlphaZeroAgent on device: {self.device}")
        
        # Enable cuDNN benchmarking for faster training
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def choose_action(self, env : SplendorLightZeroEnv, temperature=1.0):
        """
        Choose the best action based on the current state using MCTS
        Args:
            state: The current game state
            temperature: Controls exploration vs exploitation. Higher values mean more exploration
        Returns:
            action: The chosen action
            pi: The policy distribution over actions
        """
        # Run MCTS to get policy distribution
        pi = self.mcts.run(env)
        
        # Apply temperature to action probabilities
        if temperature != 1.0:
            pi = np.power(pi, 1.0/temperature)
            pi = pi / pi.sum()
        
        # Choose action based on policy distribution
        action = np.random.choice(len(pi), p=pi)
        
        return action, pi

    def get_best_action(self, state, env=None):
        """
        Get the best action based on policy and value predictions
        Args:
            state: The game state
            env: The environment instance (optional) to get action mask
        Returns:
            best_action: The action with highest policy probability
            policy: Policy distribution over actions
            value: Predicted value of the state
        """
        self.net.eval()
        
        with torch.no_grad():
            # Use non-blocking transfer if enabled
            non_blocking = self.config.get('non_blocking', True)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device, non_blocking=non_blocking)
            policy, value = self.net(state_tensor)
            policy = policy.squeeze(0)
            value = value.squeeze(0)
            
            # Get action mask if environment is provided
            if env is not None:
                action_mask = torch.FloatTensor(env.get_action_mask()).to(self.device)
                # Mask illegal actions by setting their probabilities to 0
                policy = policy * action_mask
                if policy.sum() > 0:
                    policy = policy / policy.sum()  # Renormalize
            
            # Get action with highest probability
            best_action = policy.argmax().item()
            
        return best_action

    def train(self, batch):
        """
        Train the neural network on a batch of data
        Args:
            batch: List of tuples (state, policy_target, value_target)
        Returns:
            policy_loss: Average policy loss
            value_loss: Average value loss
            total_loss: Average total loss
        """
        self.net.train()
        total_p_loss = 0
        total_v_loss = 0
        total_loss = 0
        
        # Process batch
        obs_batch, pi_batch, z_batch = [], [], []
        for st, pi_v, combined_z in batch:
            z = combined_z[0] if isinstance(combined_z, tuple) else combined_z
            obs_batch.append(st)
            pi_batch.append(pi_v)
            z_batch.append(z)
        
        # Convert to tensors and move to device with non-blocking transfer
        non_blocking = self.config.get('non_blocking', True)
        x = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(self.device, non_blocking=non_blocking)
        target_pi = torch.tensor(np.array(pi_batch), dtype=torch.float32).to(self.device, non_blocking=non_blocking)
        target_v = torch.tensor(np.array(z_batch), dtype=torch.float32).to(self.device, non_blocking=non_blocking)
        
        # Forward pass
        pred_pi, pred_v = self.net(x)
        
        # Calculate losses
        loss_p = - (target_pi * torch.log(pred_pi + 1e-8)).sum(dim=1).mean()
        loss_v = (pred_v - target_v).pow(2).mean()
        loss = loss_p + loss_v
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear cache after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log training progress
        if self.config.get('debug_mode', False):
            self.logger.info(f"Training - Policy Loss: {loss_p.item():.4f}, "
                           f"Value Loss: {loss_v.item():.4f}, "
                           f"Total Loss: {loss.item():.4f}")
        
        return loss_p.item(), loss_v.item(), loss.item()

    def save(self, path):
        """Save the agent's network weights"""
        torch.save({
            'network_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device)
        }, path)

    def load(self, path):
        """Load the agent's network weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['network_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 