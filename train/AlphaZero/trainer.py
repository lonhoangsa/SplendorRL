# Trainer for AlphaZero
import os
import numpy as np
import torch
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from env.splendor_lightzero_env import SplendorLightZeroEnv
from train.AlphaZero.agent import AlphaZeroAgent
from train.AlphaZero.buffer import ReplayBuffer
from train.AlphaZero.config import default_config
import sys

# Setup logging
def setup_logging(debug=False):
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'alphazero_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class AlphaZeroTrainer:
    def __init__(self, config=None, debug=False):
        self.logger = setup_logging(debug=debug)
        self.logger.info("Initializing AlphaZero trainer")
        
        self.cfg = default_config if config is None else config
        self.cfg['debug_mode'] = debug
        
        self.logger.info(f"Debug mode: {self.cfg['debug_mode']}")
        
        self.env = SplendorLightZeroEnv({'battle_mode':'self_play_mode'})
        
        # Extract dimensions from the environment
        card_feature_dim = self.env.primary_cards.shape[1]
        state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
        action_dim = self.env.output_nodes
        
        # Initialize agent
        self.agent = AlphaZeroAgent(state_dim, action_dim, self.cfg)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.agent.net.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(self.cfg['replay_size'])
        
        # Lists to store metrics history
        self.policy_loss_history = []
        self.value_loss_history = []
        self.total_loss_history = []
        self.iteration_numbers = []
        self.evaluation_scores = []

        self.logger.info(f"Environment initialized with state_dim={state_dim}, action_dim={action_dim}")
        self.logger.info(f"Training parameters: lr={self.cfg['lr']}, weight_decay={self.cfg['weight_decay']}, "
                        f"batch_size={self.cfg['batch_size']}, replay_size={self.cfg['replay_size']}, "
                        f"num_simulations={self.cfg['num_simulations']}")
        self.logger.info("=" * 50)

    def _check_set_completion(self, env, player_idx):
        """
        Check if a player has completed a set of cards (3 or more cards of the same color)
        Args:
            env: The game environment
            player_idx: Index of the player to check
        Returns:
            bool: True if player has completed a set, False otherwise
        """
        player = env.players[player_idx]
        color_counts = {}
        
        # Count cards of each color
        for card in player['cards']:
            color = card['color']
            color_counts[color] = color_counts.get(color, 0) + 1
        
        # Check if any color has 3 or more cards
        return any(count >= 3 for count in color_counts.values())

    def _calculate_efficiency(self, env, player_idx):
        """
        Calculate how efficiently a player is using their resources
        Args:
            env: The game environment
            player_idx: Index of the player to check
        Returns:
            float: Efficiency score between 0 and 1
        """
        player = env.players[player_idx]
        total_resources = sum(player['tokens'].values())
        total_cards = sum(player['cards'].values())
        
        # Resources are more valuable when player has fewer cards
        # Add 1 to both to avoid division by zero
        efficiency = (total_resources + 1) / (total_cards + 1)
        
        # Normalize to range [0, 1]
        return min(efficiency, 1.0)

    def self_play(self):
        self.logger.info("Starting self-play phase")
        game_rewards = []
        game_lengths = []
        
        for game_idx in range(self.cfg['games_per_iteration']):
            states, pis, players = [], [], []
            env = SplendorLightZeroEnv({
                'battle_mode': 'self_play_mode'
            })
            state_dict = env.reset()
            obs = np.concatenate([
                state_dict['observation']['tier1'].flatten(), 
                state_dict['observation']['tier2'].flatten(), 
                state_dict['observation']['tier3'].flatten(), 
                state_dict['observation']['tokens'], 
                state_dict['observation']['current_player'], 
                state_dict['observation']['nobles'].flatten()
            ])
            done = False
            steps = 0
            
            # Track total rewards for each player
            total_env_rewards = [0] * env.num_agents
            noble_counts = [0] * env.num_agents
            set_counts = [0] * env.num_agents
            efficiency_scores = [0] * env.num_agents
            
            self.logger.info(f"Starting game {game_idx+1}/{self.cfg['games_per_iteration']}")
            
            while not done:
                # Adjust temperature for action selection
                temperature = max(self.cfg['temperature'] * (self.cfg['temperature_decay'] ** game_idx), 
                                self.cfg['min_temperature'])
                
                # Use agent to choose action
                action, pi = self.agent.choose_action(env, temperature)
                
                states.append(obs)
                pis.append(pi)
                current_player = env.current_player_index
                players.append(current_player)
                
                self.logger.debug(f"Game {game_idx+1}, Step {steps+1}: Player {current_player} chose action {action}")
                
                next_state_dict = env.step(action)
                env_reward = next_state_dict.reward
                
                # Calculate additional rewards
                additional_reward = 0
                
                # Noble tile bonus
                if hasattr(next_state_dict, 'info') and 'noble_acquired' in next_state_dict.info:
                    noble_counts[current_player] += 1
                    additional_reward += self.cfg['noble_bonus']
                
                # Set bonus (for collecting cards of same color)
                if hasattr(next_state_dict, 'info') and 'card_acquired' in next_state_dict.info:
                    if self._check_set_completion(env, current_player):
                        set_counts[current_player] += 1
                        additional_reward += self.cfg['set_bonus']
                
                # Efficiency bonus (for using resources efficiently)
                efficiency = self._calculate_efficiency(env, current_player)
                efficiency_scores[current_player] = efficiency
                additional_reward += efficiency * self.cfg['efficiency_bonus']
                
                # Update total reward with additional bonuses
                total_env_rewards[current_player] += env_reward + additional_reward
                
                # Update next observation
                next_obs = np.concatenate([
                    next_state_dict.obs['tier1'].flatten(), 
                    next_state_dict.obs['tier2'].flatten(), 
                    next_state_dict.obs['tier3'].flatten(), 
                    next_state_dict.obs['tokens'], 
                    next_state_dict.obs['current_player'], 
                    next_state_dict.obs['nobles'].flatten()
                ])
                
                done = next_state_dict.done
                obs = next_obs
                steps += 1
            
            game_lengths.append(steps)
            
            # Get final scores and create combined rewards
            scores = [p['score'] for p in env.players]
            
            # Debug logging
            if self.cfg['debug_mode'] or game_idx == 0:
                self.logger.info("Player performance summary:")
                for i, player in enumerate(env.players):
                    self.logger.info(f"Player {i}: Score={player['score']}, "
                                   f"Total Reward={total_env_rewards[i]}, "
                                   f"Nobles={noble_counts[i]}, "
                                   f"Sets={set_counts[i]}, "
                                   f"Efficiency={efficiency_scores[i]}")
            
            combined_rewards = [(scores[i], total_env_rewards[i]) for i in range(env.num_agents)]
            game_rewards.append(combined_rewards)
            
            # Determine winners based on score
            max_score = max(scores)
            winners = np.array([i for i, score in enumerate(scores) if score == max_score])
            
            self.logger.info(f"Game {game_idx+1} completed in {steps} steps.")
            self.logger.info(f"Scores: {scores}, Total Env Rewards: {total_env_rewards}, Winners: {winners}")
            
            for st, pi_v, pl in zip(states, pis, players):
                z = 1 if pl in winners else -1
                combined_z = (z, combined_rewards[pl])
                self.buffer.push((st, pi_v, combined_z))
        
        avg_steps = np.mean(game_lengths)
        avg_scores = np.mean([[r[0] for r in rewards] for rewards in game_rewards], axis=0)
        avg_total_rewards = np.mean([[r[1] for r in rewards] for rewards in game_rewards], axis=0)
        
        self.logger.info(f"Self-play phase completed. Average steps: {avg_steps:.2f}")
        self.logger.info(f"Average scores: {avg_scores}, Average total rewards: {avg_total_rewards}")

    def evaluate(self, num_games=10):
        """
        Evaluate the agent by playing against itself
        Args:
            num_games: Number of games to play for evaluation
        Returns:
            avg_score: Average score across all games
            win_rate: Win rate of the agent
        """
        self.logger.info(f"Starting evaluation phase with {num_games} games")
        scores = []
        wins = 0
        
        for game_idx in range(num_games):
            env = SplendorLightZeroEnv({'battle_mode': 'self_play_mode'})
            state_dict = env.reset()
            obs = np.concatenate([
                state_dict['observation']['tier1'].flatten(), 
                state_dict['observation']['tier2'].flatten(), 
                state_dict['observation']['tier3'].flatten(), 
                state_dict['observation']['tokens'], 
                state_dict['observation']['current_player'], 
                state_dict['observation']['nobles'].flatten()
            ])
            done = False
            
            while not done:
                # Use agent to choose action with low temperature for evaluation
                action, _ = self.agent.choose_action(env, temperature=0.1)
                
                next_state_dict = env.step(action)
                obs = np.concatenate([
                    next_state_dict.obs['tier1'].flatten(), 
                    next_state_dict.obs['tier2'].flatten(), 
                    next_state_dict.obs['tier3'].flatten(), 
                    next_state_dict.obs['tokens'], 
                    next_state_dict.obs['current_player'], 
                    next_state_dict.obs['nobles'].flatten()
                ])
                done = next_state_dict.done
            
            # Get final score
            final_score = env.players[0]['score']
            scores.append(final_score)
            
            # Check if agent won
            if final_score == max(p['score'] for p in env.players):
                wins += 1
            
            self.logger.info(f"Evaluation game {game_idx+1}/{num_games} completed. Score: {final_score}")
        
        avg_score = np.mean(scores)
        win_rate = wins / num_games
        
        self.logger.info(f"Evaluation completed. Average score: {avg_score:.2f}, Win rate: {win_rate:.2%}")
        self.evaluation_scores.append(avg_score)
        
        return avg_score, win_rate

    def train(self):
        self.logger.info("Starting training phase")
        if len(self.buffer) < self.cfg['batch_size']:
            self.logger.warning(f"Buffer size ({len(self.buffer)}) less than batch size ({self.cfg['batch_size']}). Skipping training.")
            return None, None, None
        
        total_p_loss = 0
        total_v_loss = 0
        total_loss = 0
        
        self.agent.net.train()
        for epoch in range(self.cfg['epochs']):
            batch = self.buffer.sample(self.cfg['batch_size'])
            obs_batch, pi_batch, z_batch = [], [], []
            
            for st, pi_v, combined_z in batch:
                z = combined_z[0] if isinstance(combined_z, tuple) else combined_z
                obs_batch.append(st)
                pi_batch.append(pi_v)
                z_batch.append(z)
            
            # Convert to tensors
            x = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(self.agent.device)
            target_pi = torch.tensor(np.array(pi_batch), dtype=torch.float32).to(self.agent.device)
            target_v = torch.tensor(np.array(z_batch), dtype=torch.float32).to(self.agent.device)
            
            # Forward pass
            pred_pi, pred_v = self.agent.net(x)
            
            # Calculate losses
            loss_p = - (target_pi * torch.log(pred_pi + 1e-8)).sum(dim=1).mean()
            loss_v = (pred_v - target_v).pow(2).mean()
            loss = loss_p + loss_v
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_p_loss += loss_p.item()
            total_v_loss += loss_v.item()
            total_loss += loss.item()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.cfg['epochs']}, "
                                f"Policy Loss: {loss_p.item():.4f}, "
                                f"Value Loss: {loss_v.item():.4f}, "
                                f"Total Loss: {loss.item():.4f}")
        
        # Calculate average losses
        avg_p_loss = total_p_loss / self.cfg['epochs']
        avg_v_loss = total_v_loss / self.cfg['epochs']
        avg_loss = total_loss / self.cfg['epochs']
        
        self.logger.info(f"Training completed. "
                        f"Average Policy Loss: {avg_p_loss:.4f}, "
                        f"Average Value Loss: {avg_v_loss:.4f}, "
                        f"Average Total Loss: {avg_loss:.4f}")
        
        return avg_p_loss, avg_v_loss, avg_loss

    def save_model(self, path='models/alphazero.pth'):
        model_dir = os.path.dirname(path)
        os.makedirs(model_dir, exist_ok=True)
        torch.save({
            'network_state_dict': self.agent.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': len(self.iteration_numbers),
            'policy_loss_history': self.policy_loss_history,
            'value_loss_history': self.value_loss_history,
            'total_loss_history': self.total_loss_history,
            'iteration_numbers': self.iteration_numbers,
            'evaluation_scores': self.evaluation_scores,
            'buffer': self.buffer.buffer if hasattr(self.buffer, 'buffer') else []
        }, path)
        self.logger.info(f"Model and training state saved at {path}")

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            
            # Load network state dict
            self.agent.net.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training history
            if 'iteration' in checkpoint:
                iteration = checkpoint['iteration']
                self.policy_loss_history = checkpoint.get('policy_loss_history', [])
                self.value_loss_history = checkpoint.get('value_loss_history', [])
                self.total_loss_history = checkpoint.get('total_loss_history', [])
                self.iteration_numbers = checkpoint.get('iteration_numbers', [])
                self.evaluation_scores = checkpoint.get('evaluation_scores', [])
                
                if hasattr(checkpoint, 'buffer'):
                    self.buffer.buffer = checkpoint['buffer']
                
                self.logger.info(f"Loaded model from iteration {iteration}")
                return iteration
            else:
                self.logger.info("No iteration information found in checkpoint")
                return 0
        else:
            self.logger.warning(f"Model file {path} not found. Starting with a new model.")
            return 0

    def plot_loss_history(self, save_path=None):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.iteration_numbers, self.policy_loss_history, 'r-')
        plt.title('Policy Loss over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Policy Loss')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(self.iteration_numbers, self.value_loss_history, 'g-')
        plt.title('Value Loss over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Value Loss')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(self.iteration_numbers, self.total_loss_history, 'b-')
        plt.title('Total Loss over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            if not self.cfg['debug_mode']:
                self.logger.info(f"Loss history plots saved to {save_path}")
        
        if not self.cfg['debug_mode']:
            plt.close()
        else:
            plt.show()

    def plot_evaluation_scores(self, save_path=None):
        if not self.evaluation_scores:
            self.logger.warning("No evaluation scores to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.evaluation_scores) + 1), self.evaluation_scores, 'b-')
        plt.title('Evaluation Scores over Time')
        plt.xlabel('Evaluation')
        plt.ylabel('Average Score')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            if not self.cfg['debug_mode']:
                self.logger.info(f"Evaluation scores plot saved to {save_path}")
        
        if not self.cfg['debug_mode']:
            plt.close()
        else:
            plt.show()

    def run(self, iterations=100, load_path=None, save_interval=1, continue_training=True, eval_interval=5):
        # Create directories for models and plots
        model_dir = 'models'
        plots_dir = 'plots'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Load existing model if specified and get starting iteration
        start_iteration = 0
        if load_path:
            start_iteration = self.load_model(load_path)
            if not continue_training:
                start_iteration = 0
                self.iteration_numbers = []
                self.policy_loss_history = []
                self.value_loss_history = []
                self.total_loss_history = []
                self.evaluation_scores = []
                self.logger.info("Ignoring previous training progress, starting from iteration 0")
        
        self.logger.info(f"Starting AlphaZero training from iteration {start_iteration+1} for {iterations} iterations")
        
        for i in range(start_iteration, start_iteration + iterations):
            self.logger.info(f"Starting iteration {i+1}/{start_iteration + iterations}")
            
            # Self-play phase
            self.self_play()
            
            # Training phase
            p_loss, v_loss, total_loss = self.train()
            if p_loss is not None:
                self.policy_loss_history.append(p_loss)
                self.value_loss_history.append(v_loss)
                self.total_loss_history.append(total_loss)
                self.iteration_numbers.append(i+1)
            
            # Evaluation phase
            if (i+1) % eval_interval == 0:
                avg_score, win_rate = self.evaluate()
                self.logger.info(f"Evaluation results - Average score: {avg_score:.2f}, Win rate: {win_rate:.2%}")
            
            # Save model at specified intervals
            if (i+1) % save_interval == 0:
                model_path = os.path.join(model_dir, 'alphazero_final.pth')
                self.save_model(model_path)
                
                # Generate and save plots
                if len(self.iteration_numbers) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    loss_plot_path = os.path.join(plots_dir, f'alphazero_loss_history_{timestamp}.png')
                    eval_plot_path = os.path.join(plots_dir, f'alphazero_evaluation_{timestamp}.png')
                    self.plot_loss_history(save_path=loss_plot_path)
                    self.plot_evaluation_scores(save_path=eval_plot_path)
        
        # Save final model
        final_model_path = os.path.join(model_dir, 'alphazero_final.pth')
        self.save_model(final_model_path)
        
        # Generate final plots
        if len(self.iteration_numbers) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_loss_plot_path = os.path.join(plots_dir, f'alphazero_final_loss_history_{timestamp}.png')
            final_eval_plot_path = os.path.join(plots_dir, f'alphazero_final_evaluation_{timestamp}.png')
            self.plot_loss_history(save_path=final_loss_plot_path)
            self.plot_evaluation_scores(save_path=final_eval_plot_path)
        
        self.logger.info(f"AlphaZero training completed. Final model saved at {final_model_path}")
                
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train AlphaZero agent')
    parser.add_argument('--iterations', type=int, default=50, help='Number of training iterations')
    parser.add_argument('--load_path', type=str, default='models/alphazero_final.pth', help='Path to load model from')
    parser.add_argument('--save_interval', type=int, default=10, help='Number of iterations between model saves')
    parser.add_argument('--eval_interval', type=int, default=5, help='Number of iterations between evaluations')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no_continue', action='store_true', help='Start training from scratch even if loading a model')
    args = parser.parse_args()
    
    trainer = AlphaZeroTrainer(debug=args.debug)
    trainer.run(iterations=args.iterations, 
                load_path=args.load_path, 
                save_interval=args.save_interval,
                eval_interval=args.eval_interval,
                continue_training=not args.no_continue)