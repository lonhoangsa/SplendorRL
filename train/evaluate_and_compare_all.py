import numpy as np
import torch
import os
import logging
from datetime import datetime
from env.splendor_lightzero_env import SplendorLightZeroEnv
from train.DQN.DQN_Agent import DQNAgent
from train.DuelingDQN.DuelingDQN_Agent import DuelingDQNAgent
from train.AlphaZero.agent import AlphaZeroAgent
from train.AlphaZero.config import default_config

def setup_logging():
    log_dir = '../splendor/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'evaluate_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def evaluate_models_against_each_other(env, agents, num_episodes, logger, model_names):
    logger.info(f"Starting evaluation of {', '.join(model_names)}")
    
    # Disable exploration for all agents
    for agent in agents:
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0
    
    wins = [0] * len(agents)
    total_scores = [[] for _ in range(len(agents))]
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
        done = False
        step_count = 0
        player_steps = [0] * len(agents)
        round_count = 0
        
        logger.info(f"\n=== Starting Game {valid_episodes + 1} ===")
        
        while not done and step_count < env.max_episode_steps:
            action_mask = state_dict['action_mask']
            current_player = state_dict['to_play']
            
            # Choose agent based on current player
            agent = agents[current_player]
            
            # Get action and value based on agent type
            if isinstance(agent, (DQNAgent, DuelingDQNAgent)):
                action = agent.act(state, action_mask)
                # Get Q-value for the chosen action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    q_values = agent.q_network(state_tensor)
                    value = q_values[0, action].item()
            else:  # AlphaZero agent
                action, value, _ = agent.get_best_action(state)
                # Convert numpy array to float if needed
                if isinstance(value, np.ndarray):
                    value = float(value[0])
            
            # Log action and value
            logger.info(f"Round {round_count + 1}, Player {current_player + 1} ({model_names[current_player]}):")
            logger.info(f"Action: {action}")
            logger.info(f"Value: {value:.4f}")
            
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
            
            done = next_state_dict.done
            
            # Update steps and round count
            player_steps[current_player] += 1
            if next_state_dict.obs['to_play'] == 0:  # New round starts
                round_count += 1
                logger.info(f"End of Round {round_count}")
                logger.info(f"Current scores: {[p['score'] for p in env.players]}")
            
            state_dict = next_state_dict.obs
            step_count += 1

        if step_count == env.max_episode_steps:
            logger.info(f"Game reached {step_count} steps (max limit), discarding and re-evaluating.")
            continue
            
        # Get final scores from environment
        final_scores = [p['score'] for p in env.players]
        winner = np.argmax(final_scores)
        
        if winner is not None and winner < len(agents):
            winner_steps.append(player_steps[winner])
            wins[winner] += 1
            for i in range(len(agents)):
                total_scores[i].append(final_scores[i])
            
            logger.info(f"\nGame {valid_episodes+1} Results:")
            logger.info(f"Final Scores: {final_scores}")
            logger.info(f"Winner Steps: {player_steps[winner]}")
            logger.info(f"Winner: {model_names[winner]}")
            valid_episodes += 1

    avg_scores = [np.mean(s) for s in total_scores]
    avg_winner_steps = np.mean(winner_steps) if winner_steps else 0
    
    logger.info(f"\nEvaluation completed:")
    for i, name in enumerate(model_names):
        logger.info(f"{name}: Avg Score: {avg_scores[i]:.2f}, Wins: {wins[i]}")
    logger.info(f"Average Winner Steps: {avg_winner_steps:.2f}")
    
    return avg_scores, avg_winner_steps, wins

def compare_all_models():
    logger = setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize environment with 4 players
    env = SplendorLightZeroEnv({'battle_mode': 'self_play_mode', 'num_agents': 4})
    card_feature_dim = env.primary_cards.shape[1]
    state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
    action_dim = env.output_nodes

    # Initialize all agents
    alphazero_agent = AlphaZeroAgent(state_dim, action_dim, default_config)
    dqn_agent1 = DQNAgent(state_dim, action_dim, device=device, logger=logger)
    dqn_agent2 = DQNAgent(state_dim, action_dim, device=device, logger=logger)
    dueling_dqn_agent = DuelingDQNAgent(state_dim, action_dim, device=device, logger=logger)

    # Load model weights
    model_paths = {
        'AlphaZero': '../splendor/models/alphazero_final.pth',
        'DQN1': '../splendor/models/dqn_model_final.pth',
        'DQN2': '../splendor/models/dqn_model_final.pth',
        'DuelingDQN': '../splendor/models/dueling_model_final.pth'
    }

    agents = []
    model_names = []
    
    # Load models in the specified order
    for name, path in model_paths.items():
        if os.path.exists(path):
            if name == 'AlphaZero':
                checkpoint = torch.load(path, map_location=device)
                alphazero_agent.net.load_state_dict(checkpoint['network_state_dict'])
                agents.append(alphazero_agent)
                model_names.append(name)
            elif name == 'DQN1':
                checkpoint = torch.load(path, map_location=device)
                dqn_agent1.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                dqn_agent1.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                agents.append(dqn_agent1)
                model_names.append(name)
            elif name == 'DQN2':
                checkpoint = torch.load(path, map_location=device)
                dqn_agent2.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                dqn_agent2.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                agents.append(dqn_agent2)
                model_names.append(name)
            elif name == 'DuelingDQN':
                checkpoint = torch.load(path, map_location=device)
                dueling_dqn_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                dueling_dqn_agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                agents.append(dueling_dqn_agent)
                model_names.append(name)
            logger.info(f"Loaded {name} model weights from {path}")
        else:
            logger.error(f"{name} model file {path} not found.")

    if len(agents) < 4:
        logger.error("Not all models were loaded successfully. Aborting comparison.")
        return

    # Run evaluation
    num_episodes = 100
    logger.info("\n=== Starting Model Comparison ===")
    logger.info("Player order: 1. AlphaZero, 2. DQN1, 3. DQN2, 4. DuelingDQN")
    
    avg_scores, avg_winner_steps, wins = evaluate_models_against_each_other(
        env, agents, num_episodes, logger, model_names
    )

    # Print detailed results
    logger.info("\n=== Final Results ===")
    for i, name in enumerate(model_names):
        win_rate = wins[i] / num_episodes
        logger.info(f"{name}:")
        logger.info(f"- Average Score: {avg_scores[i]:.2f}")
        logger.info(f"- Wins: {wins[i]} ({win_rate:.2%})")
    
    logger.info(f"\nAverage Winner Steps: {avg_winner_steps:.2f}")

if __name__ == "__main__":
    compare_all_models() 