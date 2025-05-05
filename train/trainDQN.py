import random
import numpy as np
import torch
import os
import logging
import matplotlib.pyplot as plt  # Thêm import matplotlib
from datetime import datetime
from env.splendor_lightzero_env import SplendorLightZeroEnv  # Import từ package env
from train.DQN.DQN_Agent import DQNAgent  # Import từ cùng package train

# Setup logging
def setup_logging():
    log_dir = '../splendor/logs'  # Điều chỉnh đường dẫn lên thư mục gốc
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'splendor_dqn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Hàm đánh giá hiệu năng của mô hình
def evaluate_dqn(env, agent, num_episodes, logger):
    logger.info("Starting evaluation of trained DQN model")
    agent.epsilon = 0  # Tắt khám phá ngẫu nhiên trong chế độ đánh giá
    total_rewards = [[] for _ in range(env.num_agents)]  # Tổng reward cho từng người chơi
    total_steps = []
    wins = [0] * env.num_agents  # Số lần thắng cho từng người chơi

    for episode in range(num_episodes):
        state_dict = env.reset()
        state = np.concatenate([state_dict['observation']['tier1'].flatten(), 
                                state_dict['observation']['tier2'].flatten(), 
                                state_dict['observation']['tier3'].flatten(), 
                                state_dict['observation']['tokens'], 
                                state_dict['observation']['current_player'], 
                                state_dict['observation']['nobles'].flatten()])
        episode_reward = [0] * env.num_agents
        done = False
        step_count = 1

        while not done and step_count < env.max_episode_steps:
            action_mask = state_dict['action_mask']
            current_player = state_dict['to_play']
            action = agent.act(state, action_mask)
            logger.info(f"Player {current_player} chose action {action}")  # Debug hành động
            next_state_dict = env.step(action)
            next_state = np.concatenate([next_state_dict.obs['tier1'].flatten(), 
                                         next_state_dict.obs['tier2'].flatten(), 
                                         next_state_dict.obs['tier3'].flatten(), 
                                         next_state_dict.obs['tokens'], 
                                         next_state_dict.obs['current_player'], 
                                         next_state_dict.obs['nobles'].flatten()])
            reward = next_state_dict.reward
            done = next_state_dict.done
            episode_reward[current_player] += reward

            state = next_state
            state_dict = next_state_dict.obs
            step_count += 1

        total_rewards = [total_rewards[i] + [episode_reward[i]] for i in range(env.num_agents)]
        total_steps.append(step_count)
        
        wins[next_state_dict.info['player']] += 1

        logger.info(f"Eval Episode {episode + 1}/{num_episodes}, Rewards: {episode_reward}, Steps: {step_count}")

    avg_rewards = [np.mean(rewards) for rewards in total_rewards]
    avg_steps = np.mean(total_steps)
    logger.info(f"Evaluation completed: Avg Rewards: {avg_rewards}, Avg Steps: {avg_steps:.2f}, Wins: {wins}")
    return avg_rewards, avg_steps, wins

def train_dqn(load_model_path=None):
    logger = setup_logging()
    logger.info("Starting DQN training for SplendorLightZeroEnv")

    env = SplendorLightZeroEnv({'battle_mode': 'self_play_mode'})
    card_feature_dim = env.primary_cards.shape[1]
    state_dim = (4 * card_feature_dim * 3) + 6 + (6 + 5 + card_feature_dim * 3) + (5 * 5)
    action_dim = env.output_nodes
    agent = DQNAgent(state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu', logger=logger)
    episodes = 5000
    save_interval = 500  # Lưu mô hình mỗi 100 episodes
    model_dir = '../splendor/models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Thêm directory để lưu plot loss
    plots_dir = '../splendor/plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Khởi tạo danh sách để lưu loss của các episode
    loss_history = []
    episode_numbers = []

    start_episode = 0  # Bắt đầu từ episode 0
    if load_model_path and os.path.exists(load_model_path):
        checkpoint = torch.load(load_model_path)
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        # Không load optimizer, epsilon hay episode để bắt đầu mới từ trọng số đã có
        logger.info(f"Loaded model weights from {load_model_path}")

    logger.info(f"Environment initialized with state_dim={state_dim}, action_dim={action_dim}")
    logger.info(f"Training parameters: episodes={episodes}, batch_size={agent.batch_size}, gamma={agent.gamma}, epsilon_start={agent.epsilon}")

    for episode in range(start_episode, episodes):
        state_dict = env.reset()
        state = np.concatenate([state_dict['observation']['tier1'].flatten(), 
                                state_dict['observation']['tier2'].flatten(), 
                                state_dict['observation']['tier3'].flatten(), 
                                state_dict['observation']['tokens'], 
                                state_dict['observation']['current_player'], 
                                state_dict['observation']['nobles'].flatten()])
        state_dim = state.shape[0]  # Lấy kích thước thực tế
        action_dim = env.output_nodes
        total_reward = [0] * env.num_agents
        total_loss = 0  # Tích lũy loss trong episode
        loss_count = 0  # Đếm số lần tính loss
        done = False
        step_count = 1

        while not done and step_count < env.max_episode_steps:
            action_mask = state_dict['action_mask']
            current_player = state_dict['to_play']
            action = agent.act(state, action_mask)
            next_state_dict = env.step(action)
            next_state = np.concatenate([next_state_dict.obs['tier1'].flatten(), 
                                         next_state_dict.obs['tier2'].flatten(), 
                                         next_state_dict.obs['tier3'].flatten(), 
                                         next_state_dict.obs['tokens'], 
                                         next_state_dict.obs['current_player'], 
                                         next_state_dict.obs['nobles'].flatten()])
            reward = next_state_dict.reward
            done = next_state_dict.done
            total_reward[current_player] += reward
            
            agent.remember(state, action, reward, next_state, done, action_mask, next_state_dict.obs['action_mask'])
            loss = agent.train()  # Lấy giá trị loss từ hàm train
            if loss is not None:  # Chỉ cộng loss nếu có giá trị
                total_loss += loss
                loss_count += 1

            state = next_state
            state_dict = next_state_dict.obs
            step_count += 1

        avg_loss = total_loss / loss_count if loss_count > 0 else 0  # Tính loss trung bình
        logger.info(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Steps: {step_count}, Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}")
        
        # Lưu lại lịch sử loss
        loss_history.append(avg_loss)
        episode_numbers.append(episode + 1)

        # Lưu mô hình định kỳ
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(model_dir, f'dqn_model_episode_{episode + 1}.pth')
            torch.save({
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode': episode + 1,
                'epsilon': agent.epsilon
            }, model_path)
            logger.info(f"Model saved at {model_path}")

    # Lưu mô hình cuối cùng sau khi huấn luyện
    final_model_path = os.path.join(model_dir, 'dqn_model_final.pth')
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode': episodes,
        'epsilon': agent.epsilon
    }, final_model_path)
    logger.info(f"Final model saved at {final_model_path}")

    # Vẽ đồ thị loss và lưu lại
    plt.figure(figsize=(10, 6))
    plt.plot(episode_numbers, loss_history, 'b-')
    plt.title('DQN Training Loss over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.grid(True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f'loss_history_{timestamp}.png')
    plt.savefig(plot_path)
    logger.info(f"Loss history plot saved to {plot_path}")
    plt.show()

    # # Chạy đánh giá sau khi huấn luyện
    # evaluate_dqn(env, agent, num_episodes=5, logger=logger)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    load_model_path = '../splendor/models/dqn_model_final.pth'  # Thay bằng đường dẫn thực tế hoặc None
    train_dqn(load_model_path=load_model_path)