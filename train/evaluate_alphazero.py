import os
import numpy as np
import torch
import logging
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from env.splendor_lightzero_env import SplendorLightZeroEnv
from train.AlphaZero.agent import AlphaZeroAgent
from train.AlphaZero.config import default_config

def setup_logging(debug=False):
    """Setup logging for the evaluation script"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'alphazero_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
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

class AlphaZeroEvaluator:
    def __init__(self, config=None, debug=False):
        """Initialize the evaluator with the given configuration"""
        self.logger = setup_logging(debug=debug)
        self.logger.info("Initializing AlphaZero evaluator")
        
        self.cfg = default_config if config is None else config
        self.cfg['debug_mode'] = debug
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize environment to get dimensions
        self.env = SplendorLightZeroEnv({'battle_mode': 'eval_mode', 'num_players': 4})
        
        # Calculate state dimension using formula from train_dueling
        card_feature_dim = self.env.primary_cards.shape[1]
        state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
        action_dim = self.env.output_nodes
        
        self.logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        
        # Initialize metrics storage
        self.results = {
            'candidate_wins': 0,
            'candidate_places': {1: 0, 2: 0, 3: 0, 4: 0},  # Track all finishing positions
            'avg_candidate_score': 0,
            'avg_steps': 0,
            'game_lengths': [],
            'player_scores': [[] for _ in range(4)],  # Scores for all 4 players
            'player_ranks': [[] for _ in range(4)]    # Rankings for all 4 players
        }
        
        # Initialize player agents dictionary
        self.player_agents = {}
    
    def load_model(self, model_path, model_id='candidate'):
        """Load a model for evaluation"""
        self.logger.info(f"Loading model {model_id} from {model_path}")
        
        # Calculate dimensions from environment
        card_feature_dim = self.env.primary_cards.shape[1]
        state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
        action_dim = self.env.output_nodes
        
        # Create agent with proper dimensions
        agent = AlphaZeroAgent(state_dim, action_dim, self.cfg)
        
        # Load model weights
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                agent.net.load_state_dict(checkpoint['network_state_dict'])
                self.logger.info(f"Successfully loaded model {model_id}")
                
                # Store in player agents dictionary
                self.player_agents[model_id] = agent
                
                return True
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                return False
        else:
            self.logger.error(f"Model file not found: {model_path}")
            return False
    
    def evaluate(self, num_games=100, player_types=None):
        """
        Evaluate in a 4-player game setup
        
        Args:
            num_games: Number of games to play
            player_types: List of 4 player types ('candidate', 'model_1', 'model_2', etc., 'random', 'bot')
                          Default is ['candidate', 'random', 'random', 'random']
        """
        # Default player types if not specified
        if player_types is None:
            player_types = ['candidate', 'bot', 'random', 'random']
        
        if len(player_types) != 4:
            self.logger.error(f"Expected 4 player types, got {len(player_types)}")
            return None
            
        # Validate that all model-based player types have loaded agents
        for i, p_type in enumerate(player_types):
            if p_type not in ['random', 'bot'] and p_type not in self.player_agents:
                self.logger.error(f"Player {i} type '{p_type}' has no loaded agent")
                return None
        
        self.logger.info(f"Starting evaluation: {num_games} games with player types: {player_types}")
        
        # Reset metrics
        self.results = {
            'candidate_wins': 0,
            'candidate_places': {1: 0, 2: 0, 3: 0, 4: 0},
            'avg_candidate_score': 0,
            'avg_steps': 0,
            'game_lengths': [],
            'player_scores': [[] for _ in range(4)],
            'player_ranks': [[] for _ in range(4)]
        }
        
        total_steps = 0
        candidate_idx = player_types.index('candidate')
        
        # Use tqdm for progress bar
        for game_idx in tqdm(range(num_games), desc="Evaluating"):
            # Play one game and record results
            scores, ranks, steps = self._play_game(player_types)
            
            # Update statistics
            if ranks[candidate_idx] == 1:  # Candidate won (rank 1)
                self.results['candidate_wins'] += 1
                
            # Track candidate's placement
            self.results['candidate_places'][ranks[candidate_idx]] += 1
                
            total_steps += steps
            
            self.results['game_lengths'].append(steps)
            
            # Record scores and ranks for all players
            for i in range(4):
                self.results['player_scores'][i].append(scores[i])
                self.results['player_ranks'][i].append(ranks[i])
            
            # Log progress periodically
            if (game_idx + 1) % 10 == 0:
                win_rate = self.results['candidate_wins'] / (game_idx + 1)
                self.logger.info(f"Progress: {game_idx+1}/{num_games}, Candidate win rate: {win_rate:.4f}")
        
        # Calculate final statistics
        self.results['avg_steps'] = total_steps / num_games
        self.results['avg_candidate_score'] = sum(self.results['player_scores'][candidate_idx]) / num_games
        
        # Log final results
        self._log_results(player_types)
        
        return self.results
    
    def _play_game(self, player_types):
        """Play a single evaluation game with 4 players"""
        env = SplendorLightZeroEnv({'battle_mode': 'eval_mode', 'num_players': 4})
        state = env.reset()
        
        steps = 0
        done = False
        
        self.logger.debug(f"Starting game with player types: {player_types}")
        
        while not done:
            current_player = env.current_player_index
            current_type = player_types[current_player]
            
            # Get action based on player type
            if current_type == 'random':
                action = env.random_action()
            elif current_type == 'bot':
                action = env.bot_action()
            else:
                # Use the appropriate agent for this player
                agent = self.player_agents[current_type]
                # Get current state
                obs = np.concatenate([
                    state['observation']['tier1'].flatten(), 
                    state['observation']['tier2'].flatten(), 
                    state['observation']['tier3'].flatten(), 
                    state['observation']['tokens'], 
                    state['observation']['current_player'], 
                    state['observation']['nobles'].flatten()
                ])
                # Get best action from agent
                action, _, _ = agent.get_best_action(obs)
            
            # print(f"current_player: {current_player}, Action: {action}")
            # Take the action
            next_state = env.step(action)
            done = next_state.done
            steps += 1
            
            # Prevent infinite games
            if steps >= self.cfg.get('max_eval_steps', 500):
                self.logger.warning(f"Game reached max steps ({steps}), terminating")
                break
        
        # Get final scores and calculate ranks
        scores = [p['score'] for p in env.players]
        
        # Calculate ranks (1 = highest score, 4 = lowest score)
        # Handle ties by giving the same rank
        sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
        ranks = [0, 0, 0, 0]
        
        current_rank = 1
        prev_score = None
        
        for idx in sorted_indices:
            if prev_score is not None and scores[idx] < prev_score:
                current_rank += 1
            ranks[idx] = current_rank
            prev_score = scores[idx]
        
        self.logger.debug(f"Game ended after {steps} steps. Scores: {scores}, Ranks: {ranks}")
        
        return scores, ranks, steps
    
    def _log_results(self, player_types):
        """Log the evaluation results"""
        total_games = sum(self.results['candidate_places'].values())
        win_rate = self.results['candidate_wins'] / total_games if total_games > 0 else 0
        candidate_idx = player_types.index('candidate')
        
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"Games played: {total_games}")
        self.logger.info(f"Candidate model (Player {candidate_idx}):")
        self.logger.info(f"Wins: {self.results['candidate_wins']} ({win_rate:.4f})")
        self.logger.info(f"Placements: 1st: {self.results['candidate_places'][1]}, "
                         f"2nd: {self.results['candidate_places'][2]}, "
                         f"3rd: {self.results['candidate_places'][3]}, "
                         f"4th: {self.results['candidate_places'][4]}")
        self.logger.info(f"Average candidate score: {self.results['avg_candidate_score']:.2f}")
        self.logger.info(f"Average game length: {self.results['avg_steps']:.2f} steps")
        
        # Log statistics for all player types
        for i, p_type in enumerate(player_types):
            avg_score = sum(self.results['player_scores'][i]) / total_games
            avg_rank = sum(self.results['player_ranks'][i]) / total_games
            self.logger.info(f"Player {i} ({p_type}): Avg score: {avg_score:.2f}, Avg rank: {avg_rank:.2f}")
            
        self.logger.info("=" * 50)
    
    def plot_results(self, player_types, save_path=None):
        """Plot the evaluation results for 4 players"""
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot candidate placements pie chart
        placements = [
            self.results['candidate_places'][1], 
            self.results['candidate_places'][2],
            self.results['candidate_places'][3],
            self.results['candidate_places'][4]
        ]
        axs[0, 0].pie(
            placements,
            labels=['1st Place', '2nd Place', '3rd Place', '4th Place'],
            autopct='%1.1f%%',
            explode=(0.05, 0, 0, 0),
            colors=['gold', 'silver', '#CD7F32', 'gray'],
            shadow=True
        )
        axs[0, 0].set_title('Candidate Model Placements')
        
        # Plot score comparison for all players
        all_scores = self.results['player_scores']
        axs[0, 1].boxplot(all_scores)
        axs[0, 1].set_xticklabels([f"P{i}\n({ptype})" for i, ptype in enumerate(player_types)])
        axs[0, 1].set_title('Score Distribution by Player')
        axs[0, 1].grid(True)
        
        # Plot game lengths histogram
        axs[1, 0].hist(self.results['game_lengths'], bins=20, color='skyblue', edgecolor='black')
        axs[1, 0].set_title('Game Length Distribution')
        axs[1, 0].set_xlabel('Number of Steps')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].grid(True)
        
        # Plot average ranks for each player type
        avg_ranks = [sum(ranks)/len(ranks) for ranks in self.results['player_ranks']]
        player_labels = [f"P{i}\n({ptype})" for i, ptype in enumerate(player_types)]
        colors = ['green', 'red', 'blue', 'orange']
        
        bars = axs[1, 1].bar(player_labels, avg_ranks, color=colors)
        axs[1, 1].set_title('Average Rank by Player (Lower is Better)')
        axs[1, 1].set_ylabel('Average Rank')
        axs[1, 1].set_ylim(1, 4)
        axs[1, 1].grid(True, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axs[1, 1].text(bar.get_x() + bar.get_width()/2., height - 0.1,
                    f'{height:.2f}', ha='center', va='bottom', color='white', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
        
    def compare_models(self, model_paths, num_games=50):
        """
        Compare multiple models against each other
        
        Args:
            model_paths: Dictionary of model_id -> model_path
            num_games: Number of games to play
        """
        # Load all models
        for model_id, path in model_paths.items():
            if not self.load_model(path, model_id):
                self.logger.error(f"Failed to load model {model_id}")
                return False
        
        # Setup player types based on available models
        available_models = list(self.player_agents.keys())
        
        if len(available_models) < 2:
            self.logger.error("Need at least 2 models for comparison")
            return False
        
        # Fill remaining players with random agents if needed
        player_types = available_models[:4]
        while len(player_types) < 4:
            player_types.append('bot')
        
        # Run evaluation
        results = self.evaluate(num_games, player_types)
        
        # Create plots directory if needed
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        # Save results plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f'model_comparison_{timestamp}.png')
        self.plot_results(player_types, save_path=plot_path)
        
        # Log comparison interpretation
        avg_ranks = [sum(ranks)/len(ranks) for ranks in self.results['player_ranks']]
        model_ranks = {player_types[i]: avg_ranks[i] for i in range(len(player_types))}
        
        self.logger.info("=" * 50)
        self.logger.info("MODEL COMPARISON RESULTS")
        self.logger.info("=" * 50)
        
        # Sort models by average rank (best first)
        sorted_models = sorted(model_ranks.items(), key=lambda x: x[1])
        
        for i, (model, rank) in enumerate(sorted_models):
            self.logger.info(f"Rank {i+1}: {model} (Average position: {rank:.2f})")
            
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate AlphaZero model in 4-player Splendor')
    parser.add_argument('--candidate_path', type=str, required=True, help='Path to the candidate model to evaluate')
    parser.add_argument('--player2_type', type=str, default='bot', choices=['random', 'bot', 'model'], help='Type of player 2')
    parser.add_argument('--player2_path', type=str, default=None, help='Path to player 2 model (if type is model)')
    parser.add_argument('--player3_type', type=str, default='bot', choices=['random', 'bot', 'model'], help='Type of player 3')
    parser.add_argument('--player3_path', type=str, default=None, help='Path to player 3 model (if type is model)')
    parser.add_argument('--player4_type', type=str, default='bot', choices=['random', 'bot', 'model'], help='Type of player 4')
    parser.add_argument('--player4_path', type=str, default=None, help='Path to player 4 model (if type is model)')
    parser.add_argument('--num_games', type=int, default=50, help='Number of evaluation games to play')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--plot', action='store_true', help='Plot evaluation results')
    args = parser.parse_args()
    
    evaluator = AlphaZeroEvaluator(debug=args.debug)
    
    # Load the candidate model
    if not evaluator.load_model(args.candidate_path, 'candidate'):
        return
    
    # Setup player types
    player_types = ['candidate']
    
    # Load additional models if specified
    if args.player2_type == 'model' and args.player2_path:
        if evaluator.load_model(args.player2_path, 'model_2'):
            player_types.append('model_2')
        else:
            player_types.append('random')
    else:
        player_types.append(args.player2_type)
        
    if args.player3_type == 'model' and args.player3_path:
        if evaluator.load_model(args.player3_path, 'model_3'):
            player_types.append('model_3')
        else:
            player_types.append('random')
    else:
        player_types.append(args.player3_type)
        
    if args.player4_type == 'model' and args.player4_path:
        if evaluator.load_model(args.player4_path, 'model_4'):
            player_types.append('model_4')
        else:
            player_types.append('random')
    else:
        player_types.append(args.player4_type)
    
    # Run evaluation
    results = evaluator.evaluate(args.num_games, player_types)
    
    # Plot results if requested
    if args.plot and results:
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f'evaluation_{timestamp}.png')
        evaluator.plot_results(player_types, save_path=plot_path)

if __name__ == '__main__':
    main() 