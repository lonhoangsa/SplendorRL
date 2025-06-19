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
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        self.cfg = default_config if config is None else config
        self.cfg['debug_mode'] = debug
        
        # Add early stopping parameters
        self.cfg['patience'] = 10  # Number of evaluations to wait before early stopping
        self.cfg['min_delta'] = 0.001  # Minimum change in win rate to be considered as improvement
        self.cfg['gradient_clip'] = 1.0  # Maximum gradient norm
        
        self.logger.info(f"Debug mode: {self.cfg['debug_mode']}")
        
        self.env = SplendorLightZeroEnv({'battle_mode':'self_play_mode'})
        
        # Extract dimensions from the environment
        card_feature_dim = self.env.primary_cards.shape[1]
        state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
        action_dim = self.env.output_nodes
        
        # Initialize agent with device
        self.agent = AlphaZeroAgent(state_dim, action_dim, self.cfg, device=self.device, architecture_type=self.cfg.get('architecture_type', 'simple'))
        
        # Initialize optimizer with higher initial learning rate
        self.optimizer = optim.Adam(self.agent.net.parameters(), 
                                  lr=self.cfg['lr'], 
                                  weight_decay=self.cfg['weight_decay'],
                                  betas=(0.9, 0.999))
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(self.cfg['replay_size'])
        
        # Lists to store metrics history
        self.policy_loss_history = []
        self.value_loss_history = []
        self.total_loss_history = []
        self.iteration_numbers = []
        self.evaluation_scores = []
        
        # Track best model and early stopping
        self.best_win_rate = 0
        self.best_model_state = None
        self.no_improvement_count = 0
        self.best_iteration = 0

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
                
                # Efficiency bonus
                efficiency = self._calculate_efficiency(env, current_player)
                if(action > 29): efficiency = 0
                efficiency_scores[current_player] = efficiency
                additional_reward += efficiency * 2
                
                # Speed bonus - encourage faster game completion
                if env_reward > 0:  # If player scored points
                    # Base speed bonus that decreases with more steps
                    # Starts at 10.0 and decreases by 0.3 per step, becomes 0 after ~27 steps
                    speed_bonus = max(10.0 - (steps * 0.2), 0.0)
                    additional_reward += speed_bonus
                    self.logger.debug(f"Player {current_player} received speed bonus: {speed_bonus:.2f}")
                
                # Calculate step reward
                step_reward = env_reward + additional_reward
                
                # Store step reward
                if not hasattr(self, 'step_rewards'):
                    self.step_rewards = []
                self.step_rewards.append((current_player, step_reward))
                
                # Update total reward with environment reward and efficiency bonus
                total_env_rewards[current_player] += step_reward
                
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
            
            # Check if any player has won (score >= 15)
            max_score = max(scores)
            if max_score < 15:
                # Apply heavy penalty to all players when no one wins
                penalty = -100.0  # Heavy negative reward
                self.logger.info(f"No player reached 15 points. Applying penalty of {penalty} to all players")
                for i in range(len(total_env_rewards)):
                    total_env_rewards[i] += penalty
                    # Also penalize the last few moves that led to this situation
                    if len(self.step_rewards) > 0:
                        last_moves = self.step_rewards[-3:]  # Get last 3 moves
                        for _, reward in last_moves:
                            if reward > 0:  # Only penalize positive rewards
                                reward -= abs(penalty) / 3  # Distribute penalty among last moves
            else:
                # Add final speed bonus for the winner
                winner_idx = scores.index(max_score)
                # Calculate speed bonus based on total game length
                # Starts at 15.0 and decreases by 0.5 per step, becomes 0 after 30 steps
                final_speed_bonus = max(15.0 - (steps * 0.5), 0.0)
                total_env_rewards[winner_idx] += final_speed_bonus
                self.logger.info(f"Winner (Player {winner_idx}) received final speed bonus: {final_speed_bonus:.2f}")

                # Apply speed penalties to losing players
                for i in range(len(total_env_rewards)):
                    if i != winner_idx:  # For all non-winners
                        # Calculate speed penalty that increases with game length
                        # Starts at -5.0 and decreases by 0.3 per step after 20 steps
                        speed_penalty = -5.0 if steps <= 20 else -5.0 - ((steps - 20) * 0.3)
                        total_env_rewards[i] += speed_penalty
                        self.logger.info(f"Losing player {i} received speed penalty: {speed_penalty:.2f}")
            
            combined_rewards = [(scores[i], total_env_rewards[i]) for i in range(env.num_agents)]
            game_rewards.append(combined_rewards)
            
            # Determine winners based on score
            winners = np.array([i for i, score in enumerate(scores) if score == max_score])
            
            self.logger.info(f"Game {game_idx+1} completed in {steps} steps.")
            self.logger.info(f"Scores: {scores}, Total Env Rewards: {total_env_rewards}, Winners: {winners}")
            
            # Push states, policies, and rewards to buffer
            for i, (st, pi_v, pl) in enumerate(zip(states, pis, players)):
                # Get the reward for this step
                step_reward = self.step_rewards[i][1] if i < len(self.step_rewards) else 0  # Include step reward in the combined value
                self.buffer.push((st, pi_v, step_reward))
            
            # Clear step rewards for next game
            self.step_rewards = []
        
        avg_steps = np.mean(game_lengths)
        avg_scores = np.mean([[r[0] for r in rewards] for rewards in game_rewards], axis=0)
        avg_total_rewards = np.mean([[r[1] for r in rewards] for rewards in game_rewards], axis=0)
        
        self.logger.info(f"Self-play phase completed. Average steps: {avg_steps:.2f}")
        self.logger.info(f"Average scores: {avg_scores}, Average total rewards: {avg_total_rewards}")

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
            obs_batch, pi_batch, reward_batch = [], [], []
            
            for st, pi_v, reward in batch:
                obs_batch.append(st)
                pi_batch.append(pi_v)
                reward_batch.append(reward)
            
            # Convert to tensors and move to device
            x = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(self.device)
            target_pi = torch.tensor(np.array(pi_batch), dtype=torch.float32).to(self.device)
            target_v = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(self.device)
            
            # Forward pass
            pred_pi, pred_v = self.agent.net(x)
            
            # Calculate losses
            loss_p = - (target_pi * torch.log(pred_pi + 1e-8)).sum(dim=1).mean()
            loss_v = (pred_v - target_v).pow(2).mean()
            loss = loss_p + loss_v
            
            # Update network with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.net.parameters(), self.cfg['gradient_clip'])
            self.optimizer.step()
            
            # Accumulate losses
            total_p_loss += loss_p.item()
            total_v_loss += loss_v.item()
            total_loss += loss.item()
            
            # Log progress
            if (epoch + 1) % 1000 == 0:
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

    def evaluate_models(self, model1, model2, num_games=50):
        """
        Evaluate two models against each other in a 4-player game
        Args:
            model1: First model (old model)
            model2: Second model (new model)
            num_games: Number of games to play
        Returns:
            tuple: (model1_wins, model2_wins, draws)
        """
        self.logger.info(f"Starting model evaluation: {num_games} games")
        model1_wins = 0
        model2_wins = 0
        draws = 0
        valid_games = 0  # Count of games where winner's score > 15
        
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
            
            # Alternate model assignments based on game index
            if game_idx % 2 == 0:
                # Even games: model1 for players 0,2; model2 for players 1,3
                model_assignments = {0: model1, 1: model2, 2: model1, 3: model2}
            else:
                # Odd games: model2 for players 0,2; model1 for players 1,3
                model_assignments = {0: model2, 1: model1, 2: model2, 3: model1}
            
            while not done:
                current_player = env.current_player_index
                # Get action from current model with temperature=0.1 (small but not zero)
                current_model = model_assignments[current_player]
                action = current_model.get_best_action(obs, env)
                
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
            
            # Get final scores
            scores = [p['score'] for p in env.players]
            max_score = max(scores)
            winners = [i for i, score in enumerate(scores) if score == max_score]
            
            # Only count results if winner's score > 15
            if max_score > 15:
                valid_games += 1
                # Count wins based on model assignments
                if len(winners) == 1:  # No draw
                    winner = winners[0]
                    winning_model = model_assignments[winner]
                    if winning_model == model1:
                        model1_wins += 1
                    else:
                        model2_wins += 1
                else:  # Draw
                    draws += 1
            
            self.logger.info(f"Completed {game_idx + 1}/{num_games} evaluation games")
            self.logger.info(f"winners: {winners}, scores: {max_score}")
            self.logger.info(f"Current results: Model1 wins={model1_wins}, Model2 wins={model2_wins}, Draws={draws}")
            self.logger.info(f"Valid games (score > 15): {valid_games}")
        
        # Calculate win rate based on valid games only
        if valid_games > 0:
            win_rate = (model2_wins / valid_games) * 100
            self.logger.info(f"Evaluation results: Model1 wins={model1_wins}, Model2 wins={model2_wins}, "
                           f"Draws={draws}, Model2 win rate={win_rate:.2f}%")
            self.logger.info(f"Total valid games: {valid_games} out of {num_games}")
        else:
            self.logger.warning("No valid games found (no winner scored above 15 points)")
            win_rate = 0
        
        return model1_wins, model2_wins, draws

    def save_model(self, path='models/alphazero.pth', is_checkpoint=False):
        """
        Save model only if it performs better than the previous model
        """
        model_dir = os.path.dirname(path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Add architecture type to filename
        if self.agent.net.architecture_type == 'complex':
            base_path = os.path.splitext(path)[0]
            ext = os.path.splitext(path)[1]
            path = f"{base_path}_complex{ext}"
        
        # Save current model state, including architecture type
        current_state = {
            'network_state_dict': self.agent.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'architecture_type': self.agent.net.architecture_type
        }
        
        # If there's an existing model and this is not a checkpoint, evaluate against it
        if os.path.exists(path) and not is_checkpoint:
            self.logger.info("Found existing model, evaluating new model against it...")
            
            # Calculate state dimension from observation space
            card_feature_dim = self.env.card_feature_dim
            state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
            
            # Load old model with correct dimensions and architecture
            old_model = AlphaZeroAgent(
                state_dim=state_dim,
                action_dim=self.env.output_nodes,
                config=self.cfg,
                device=self.device,
                architecture_type=self.agent.net.architecture_type  # Use same architecture type
            )
            checkpoint = torch.load(path, map_location=self.device)
            old_model.net.load_state_dict(checkpoint['network_state_dict'])
            
            # Evaluate models
            old_wins, new_wins, _ = self.evaluate_models(old_model, self.agent)
            
            win_rate = 0.0
            if (old_wins + new_wins) > 0:
                win_rate = (new_wins / (old_wins + new_wins)) * 100
            
            if win_rate > 50:
                self.logger.info(f"New model performs better (win rate: {win_rate:.2f}%). Saving...")
                torch.save(current_state, path)
                self.logger.info(f"Model saved at {path}")
                
                # Update best model if this is better
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    self.best_model_state = current_state
                    # Save best model checkpoint with architecture type in name
                    best_model_path = os.path.join(model_dir, f'alphazero_best_{self.agent.net.architecture_type}.pth')
                    torch.save(current_state, best_model_path)
                    self.logger.info(f"New best model saved at {best_model_path} with win rate {win_rate:.2f}%")
            else:
                self.logger.info(f"New model performs worse (win rate: {win_rate:.2f}%). Keeping old model.")
        else:
            # If no existing model or this is a checkpoint, save the current one
            self.logger.info("Saving current model...")
            torch.save(current_state, path)
            self.logger.info(f"Model saved at {path}")

    def load_model(self, path):
        # Add architecture type to filename if using complex architecture
        if self.agent.net.architecture_type == 'complex':
            base_path = os.path.splitext(path)[0]
            ext = os.path.splitext(path)[1]
            complex_path = f"{base_path}_complex{ext}"
        
            path = complex_path
            # self.logger.info(f"Loading complex architecture model from {path}")
            # print(path)   
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)  # Load to correct device
            
            # Get architecture type from checkpoint or use default
            architecture_type = checkpoint.get('architecture_type', 'simple')
            
            # Reinitialize network with correct architecture
            card_feature_dim = self.env.card_feature_dim
            state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
            self.agent = AlphaZeroAgent(
                state_dim=state_dim,
                action_dim=self.env.output_nodes,
                config=self.cfg,
                device=self.device,
                architecture_type=architecture_type
            )
            
            # Load network state dict
            self.agent.net.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.logger.info(f"Loaded model from {path} with architecture type: {architecture_type}")
            
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
        
        # Convert single scores to lists if necessary
        processed_scores = []
        for scores in self.evaluation_scores:
            if isinstance(scores, (int, float, np.number)):
                processed_scores.append([scores])  # Convert single score to list
            else:
                processed_scores.append(scores)
        
        # Get the maximum number of players across all evaluations
        max_players = max(len(scores) for scores in processed_scores)
        
        # Plot each player's scores
        for player_idx in range(max_players):
            # Extract scores for this player, handling cases where some evaluations might have fewer players
            player_scores = []
            for eval_scores in processed_scores:
                if player_idx < len(eval_scores):
                    player_scores.append(eval_scores[player_idx])
                else:
                    player_scores.append(None)  # Use None for missing scores
            
            # Filter out None values and plot
            valid_indices = [i for i, score in enumerate(player_scores) if score is not None]
            valid_scores = [score for score in player_scores if score is not None]
            
            if valid_scores:  # Only plot if we have valid scores
                plt.plot([i+1 for i in valid_indices], valid_scores, 
                        label=f'Player {player_idx+1}')
        
        plt.title('Evaluation Scores over Time')
        plt.xlabel('Evaluation')
        plt.ylabel('Average Score')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            if not self.cfg['debug_mode']:
                self.logger.info(f"Evaluation scores plot saved to {save_path}")
        
        if not self.cfg['debug_mode']:
            plt.close()
        else:
            plt.show()

    def run(self, iterations=100, load_path=None, save_interval=4, continue_training=True, eval_interval=5, architecture_type='simple'):
        # Create directories for models and plots
        model_dir = 'models'
        plots_dir = 'plots'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Update config with architecture type
        self.cfg['architecture_type'] = architecture_type
        
        # Reinitialize agent with new architecture if needed
        if hasattr(self, 'agent') and self.agent.net.architecture_type != architecture_type:
            card_feature_dim = self.env.card_feature_dim
            state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
            self.agent = AlphaZeroAgent(
                state_dim=state_dim,
                action_dim=self.env.output_nodes,
                config=self.cfg,
                device=self.device,
                architecture_type=architecture_type
            )
            self.optimizer = optim.Adam(self.agent.net.parameters(), 
                                      lr=self.cfg['lr'], 
                                      weight_decay=self.cfg['weight_decay'],
                                      betas=(0.9, 0.999))
        
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
            
            # Save model at specified intervals
            if (i+1) % save_interval == 0:
                # Save to final model path with architecture type
                final_model_path = os.path.join(model_dir, f'alphazero_final.pth')
                self.save_model(final_model_path)
                
                # Save checkpoint with iteration number and architecture type
                checkpoint_path = os.path.join(model_dir, f'alphazero_checkpoint_{i+1}.pth')
                self.save_model(checkpoint_path, is_checkpoint=True)
        
        # Save final model with architecture type
        final_model_path = os.path.join(model_dir, f'alphazero_final_{architecture_type}.pth')
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
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations')
    parser.add_argument('--load_path', type=str, default='models/alphazero_final.pth', help='Path to load model from')
    parser.add_argument('--save_interval', type=int, default=3, help='Number of iterations between model saves')
    parser.add_argument('--eval_interval', type=int, default=3, help='Number of iterations between evaluations')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--architecture_type', type=str, default='simple', help='Architecture type: simple or complex')
    parser.add_argument('--no_continue', action='store_true', help='Start training from scratch even if loading a model')
    args = parser.parse_args()
    
    trainer = AlphaZeroTrainer(debug=args.debug)
    trainer.run(iterations=args.iterations, 
                load_path=args.load_path, 
                save_interval=args.save_interval,
                eval_interval=args.eval_interval,
                continue_training=not args.no_continue,
                architecture_type=args.architecture_type)