import numpy as np
import torch
import os
import logging
from datetime import datetime
from env.splendor_lightzero_env import SplendorLightZeroEnv
from train.DQN_Agent import DQNAgent
from train.DuelingDQN_Agent import DuelingDQNAgent

def setup_logging():
    log_dir = '../splendor/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'evaluate_dqn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def evaluate_model(env, agent, num_episodes, logger, model_name):
    logger.info(f"Starting evaluation of trained {model_name} model")
    agent.epsilon = 0  # Disable exploration during evaluation
    total_rewards = [[] for _ in range(env.num_agents)]
    winner_steps = []
    wins = [0] * env.num_agents

    valid_episodes = 0
    while valid_episodes < num_episodes:
        state_dict = env.reset()
        state = np.concatenate([
            state_dict['observation']['tier1'].flatten(),
            state_dict['observation']['tier2'].flatten(),
            state_dict['observation']['tier3'].flatten(),
            state_dict['observation']['tokens'],
            state_dict['observation']['current_player'],
            state_dict['observation']['nobles'].flatten()
        ])
        episode_reward = [0] * env.num_agents
        done = False
        step_count = 0
        player_steps = [0] * env.num_agents
        turn = []
        
        while not done and step_count < env.max_episode_steps:
            action_mask = state_dict['action_mask']
            current_player = state_dict['to_play']
            turn.append(current_player)
            
            action = agent.act(state, action_mask)
            next_state_dict = env.step(action)
            
            # Update state for next iteration
            state = np.concatenate([
                next_state_dict.obs['tier1'].flatten(),
                next_state_dict.obs['tier2'].flatten(),
                next_state_dict.obs['tier3'].flatten(),
                next_state_dict.obs['tokens'],
                next_state_dict.obs['current_player'],
                next_state_dict.obs['nobles'].flatten()
            ])
            
            reward = next_state_dict.reward
            done = next_state_dict.done
            
            # Update rewards and steps
            episode_reward[current_player] += reward
            
            if next_state_dict.info['player'] != current_player:
                player_steps[current_player] += 1
            
            state_dict = next_state_dict.obs
            step_count += 1

        if step_count == env.max_episode_steps:
            logger.info(f"Episode reached {step_count} steps (max limit), discarding and re-evaluating.")
            continue
            
        winner = next_state_dict.info.get('player')
        winner_steps.append(player_steps[winner])
        for i in range(env.num_agents):
            total_rewards[i].append(episode_reward[i])
            
        wins[winner] += 1
        logger.info(f"Episode {valid_episodes+1} ({model_name}): Rewards {episode_reward} - Winner Steps: {player_steps[winner]} - Winner: {winner}")
        valid_episodes += 1

    avg_rewards = [np.mean(r) for r in total_rewards]
    avg_winner_steps = np.mean(winner_steps)
    logger.info(f"{model_name} Evaluation completed: Avg Rewards: {avg_rewards}, Avg Winner Steps: {avg_winner_steps:.2f}, Wins: {wins}")
    return avg_rewards, avg_winner_steps, wins

def evaluate_models_against_each_other(env, agent1, agent2, num_episodes, logger, model1_name, model2_name):
    logger.info(f"Starting evaluation of {model1_name} vs {model2_name}")
    agent1.epsilon = 0  # Disable exploration during evaluation
    agent2.epsilon = 0
    wins = [0, 0]  # [model1_wins, model2_wins]
    total_rewards = [[], []]
    winner_steps = []

    valid_episodes = 0
    while valid_episodes < num_episodes:
        state_dict = env.reset()
        state = np.concatenate([
            state_dict['observation']['tier1'].flatten(),
            state_dict['observation']['tier2'].flatten(),
            state_dict['observation']['tier3'].flatten(),
            state_dict['observation']['tokens'],
            state_dict['observation']['current_player'],
            state_dict['observation']['nobles'].flatten()
        ])
        episode_reward = [0, 0]
        done = False
        step_count = 0
        player_steps = [0, 0]  # Initialize for both players
        
        while not done and step_count < env.max_episode_steps:
            action_mask = state_dict['action_mask']
            current_player = state_dict['to_play']
            
            # Choose agent based on current player
            agent = agent1 if current_player == 0 else agent2
            action = agent.act(state, action_mask)
            next_state_dict = env.step(action)
            
            # Update state for next iteration
            state = np.concatenate([
                next_state_dict.obs['tier1'].flatten(),
                next_state_dict.obs['tier2'].flatten(),
                next_state_dict.obs['tier3'].flatten(),
                next_state_dict.obs['tokens'],
                next_state_dict.obs['current_player'],
                next_state_dict.obs['nobles'].flatten()
            ])
            
            # Handle reward properly - it's a list of rewards for each player
            rewards = next_state_dict.reward
            if isinstance(rewards, (int, float)):
                rewards = [rewards, 0] if current_player == 0 else [0, rewards]
            
            done = next_state_dict.done
            
            # Update rewards and steps
            episode_reward[0] += rewards[0]
            episode_reward[1] += rewards[1]
            
            # Update steps for the current player
            if current_player in [0, 1]:  # Ensure current_player is valid
                player_steps[current_player] += 1
            
            state_dict = next_state_dict.obs
            step_count += 1

        if step_count == env.max_episode_steps:
            logger.info(f"Episode reached {step_count} steps (max limit), discarding and re-evaluating.")
            continue
            
        winner = next_state_dict.info.get('player')
        if winner is not None and winner in [0, 1]:  # Ensure winner is valid
            winner_steps.append(player_steps[winner])
            wins[winner] += 1
            for i in range(2):
                total_rewards[i].append(episode_reward[i])
            
            logger.info(f"Episode {valid_episodes+1} ({model1_name} vs {model2_name}): "
                       f"Rewards {episode_reward} - Winner Steps: {player_steps[winner]} - Winner: {winner}")
            valid_episodes += 1

    avg_rewards = [np.mean(r) for r in total_rewards]
    avg_winner_steps = np.mean(winner_steps) if winner_steps else 0
    logger.info(f"{model1_name} vs {model2_name} Evaluation completed: "
                f"Avg Rewards: {avg_rewards}, Avg Winner Steps: {avg_winner_steps:.2f}, Wins: {wins}")
    return avg_rewards, avg_winner_steps, wins

def compare_models():
    logger = setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    env = SplendorLightZeroEnv({'battle_mode': 'self_play_mode'})
    card_feature_dim = env.primary_cards.shape[1]
    state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
    action_dim = env.output_nodes

    # Initialize both agents
    dqn_agent = DQNAgent(state_dim, action_dim, device=device, logger=logger)
    dueling_dqn_agent = DuelingDQNAgent(state_dim, action_dim, device=device, logger=logger)

    # Load the trained model weights
    dqn_model_path = '../splendor/models/dqn_model_final.pth'
    dueling_model_path = '../splendor/models/dueling_model_final.pth'

    if os.path.exists(dqn_model_path):
        checkpoint = torch.load(dqn_model_path, map_location=device)
        dqn_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        dqn_agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        logger.info(f"Loaded DQN model weights from {dqn_model_path}")
    else:
        logger.error(f"DQN model file {dqn_model_path} not found.")
        return

    if os.path.exists(dueling_model_path):
        checkpoint = torch.load(dueling_model_path, map_location=device)
        dueling_dqn_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        dueling_dqn_agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        logger.info(f"Loaded Dueling DQN model weights from {dueling_model_path}")
    else:
        logger.error(f"Dueling DQN model file {dueling_model_path} not found.")
        return

    # Evaluate both models
    num_episodes = 10
    logger.info("\n=== Starting Model Comparison ===")
    
    # Self-play evaluation
    dqn_rewards, dqn_winner_steps, dqn_wins = evaluate_model(env, dqn_agent, num_episodes, logger, "DQN")
    dueling_dqn_rewards, dueling_dqn_winner_steps, dueling_dqn_wins = evaluate_model(env, dueling_dqn_agent, num_episodes, logger, "Dueling DQN")

    # Head-to-head evaluation
    logger.info("\n=== Starting Head-to-Head Evaluation ===")
    env = SplendorLightZeroEnv({'battle_mode': 'self_play_mode', 'num_agents': 2})
    dqn_vs_dueling_rewards, dqn_vs_dueling_steps, dqn_vs_dueling_wins = evaluate_models_against_each_other(
        env, dqn_agent, dueling_dqn_agent, num_episodes, logger, "DQN", "Dueling DQN"
    )

    # Print comparison results
    logger.info("\n=== Model Comparison Results ===")
    logger.info(f"DQN Model:")
    logger.info(f"- Average Winner Steps: {dqn_winner_steps:.2f}")
    logger.info(f"- Win Distribution: {dqn_wins}")
    logger.info(f"- Average Rewards: {dqn_rewards}")
    
    logger.info(f"\nDueling DQN Model:")
    logger.info(f"- Average Winner Steps: {dueling_dqn_winner_steps:.2f}")
    logger.info(f"- Win Distribution: {dueling_dqn_wins}")
    logger.info(f"- Average Rewards: {dueling_dqn_rewards}")

    logger.info(f"\nHead-to-Head Results (DQN vs Dueling DQN):")
    logger.info(f"- Average Winner Steps: {dqn_vs_dueling_steps:.2f}")
    logger.info(f"- Win Distribution: {dqn_vs_dueling_wins}")
    logger.info(f"- Average Rewards: {dqn_vs_dueling_rewards}")

    # Calculate and print performance differences
    steps_diff = dueling_dqn_winner_steps - dqn_winner_steps
    logger.info(f"\nPerformance Comparison:")
    logger.info(f"- Steps Difference: {steps_diff:.2f} (Positive means Dueling DQN takes more steps)")
    logger.info(f"- Win Rate Difference: {[dueling_dqn_wins[i] - dqn_wins[i] for i in range(len(dqn_wins))]}")
    logger.info(f"- Head-to-Head Win Rate: DQN: {dqn_vs_dueling_wins[0]}, Dueling DQN: {dqn_vs_dueling_wins[1]}")

if __name__ == "__main__":
    compare_models()