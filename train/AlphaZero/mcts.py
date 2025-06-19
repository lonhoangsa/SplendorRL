# MCTS for AlphaZero
import copy
import math
import torch
import numpy as np
import logging
from env.splendor_lightzero_env import SplendorLightZeroEnv

class MCTSNode:
    def __init__(self, prior):
        self.P = prior  # prior probability
        self.N = 0      # visit count
        self.W = 0      # total value
        self.Q = 0      # mean value
        self.children = {}

class MCTS:
    def __init__(self, net, config):
        self.net = net
        self.num_sim = config['num_simulations']
        self.cpuct = config['cpuct']
        self.max_depth = config.get('max_depth', 500)
        # Lưu thiết bị của model
        self.device = next(net.parameters()).device
        # Khởi tạo logger
        self.logger = logging.getLogger("AlphaZero.MCTS")
        self.debug_mode = config.get('debug_mode', False)
        self.non_blocking = config.get('non_blocking', True)
        
        # Enable cuDNN benchmarking for faster inference
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def run(self, env : SplendorLightZeroEnv):
        root = MCTSNode(prior=1.0)
        obs, mask = self._encode_env(env)
        
        legal_actions_count = np.sum(mask)
        self.logger.debug(f"Legal actions: {legal_actions_count}/{len(mask)}")
        
        # Kiểm tra nếu không có hành động hợp lệ
        if legal_actions_count == 0:
            self.logger.warning("No legal actions available!")
            pi = np.ones_like(mask) / len(mask)
            return pi
            
        # Lưu trạng thái train hiện tại
        was_training = self.net.training
        self.net.eval()
        try:
            # Dự đoán prior probabilities từ mạng với non-blocking transfer
            with torch.no_grad():
                obs = obs.to(self.device, non_blocking=self.non_blocking)
                P, v = self.net(obs.unsqueeze(0))
                self.logger.debug(f"Initial value prediction: {v.item():.4f}")
                
            P = P.squeeze(0).cpu().numpy() * mask
            if P.sum() > 0:
                P = P / P.sum()
            else:
                P = mask.astype(np.float32)
                P = P / P.sum()
            
            # Khởi tạo các nút con
            for a, prob in enumerate(P):
                if mask[a]:
                    root.children[a] = MCTSNode(prior=prob)
                    if self.debug_mode:
                        self.logger.debug(f"Action {a}: prior probability = {prob:.4f}")
            
            # Thực hiện các lần mô phỏng theo 4 bước Selection, Expansion, Simulation, Backpropagation
            for i in range(self.num_sim):
                if self.debug_mode and (i+1) % 10 == 0:
                    self.logger.debug(f"Running simulation {i+1}/{self.num_sim}")
                
                # Tạo một bản sao của môi trường để thực hiện mô phỏng
                sim_env = copy.deepcopy(env)
                
                # 1. Selection: Chọn đường đi tiềm năng nhất cho đến khi gặp node lá
                path, leaf_node, leaf_env, depth = self._select(sim_env, root)
                
                # Kiểm tra nếu game đã kết thúc ở leaf node
                if hasattr(leaf_env, 'done') and leaf_env.done:
                    v = self._get_reward(leaf_env)
                else:
                    # 2. Expansion: Tạo các node con cho node lá (nếu có thể)
                    expanded = self._expand(leaf_env, leaf_node)
                    
                    # 3. Simulation: Mô phỏng đến khi game kết thúc hoặc đạt đến giới hạn độ sâu
                    v = self._simulate(leaf_env, leaf_node, expanded, depth)
                
                # 4. Backup: Cập nhật statistics cho các node trên đường đi
                self._backup(path, v)
            
            # Tính toán phân phối xác suất dựa trên số lần thăm
            pi = np.zeros_like(P)
            visit_counts = {}
            total_visits = 0
            
            for a, node in root.children.items():
                pi[a] = node.N
                visit_counts[a] = node.N
                total_visits += node.N
                
            # Đảm bảo tổng xác suất bằng 1 sau khi chuẩn hóa
            if total_visits > 0:
                pi = pi / total_visits
                if abs(pi.sum() - 1.0) > 1e-9:
                    pi = pi / pi.sum()
            else:
                self.logger.warning("No visits recorded, using uniform distribution")
                pi = mask.astype(np.float32)
                pi = pi / pi.sum()
            
            # Kiểm tra cuối cùng để đảm bảo tổng bằng 1
            assert np.isclose(pi.sum(), 1.0), f"Pi sum is {pi.sum()}, not 1.0"
            
            return pi
        finally:
            # Khôi phục lại trạng thái train ban đầu
            if was_training:
                self.net.train()
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _select(self, env, node, path=None, depth=0):
        """
        1. Selection: Chọn đường đi tiềm năng nhất cho đến khi gặp node lá hoặc node chưa được mở rộng
        """
        if path is None:
            path = []
            
        # Nếu node không có con hoặc chưa được thăm, trả về node đó
        if not node.children or node.N == 0:
            return path + [(None, node)], node, env, depth
        
        # Kiểm tra nếu có người chơi nào đã đạt điểm chiến thắng
        if self._game_is_over(env):
            return path + [(None, node)], node, env, depth
            
        # Kiểm tra giới hạn độ sâu
        if depth >= self.max_depth:
            return path + [(None, node)], node, env, depth
            
        # Chọn hành động tốt nhất theo UCB
        best_score, best_act, best_child = -float('inf'), None, None
        
        # Tính tổng N^(1/4) cho tất cả các hành động con
        sum_n_quarter = sum(child.N ** 0.25 for child in node.children.values())
        
        # Tính UCB cho mỗi hành động theo công thức: Q(s,a) + cP(s,a) * sum_b(N^(1/4)(s,b)) / (1 + N(s,a))^(1/2)
        for a, child in node.children.items():
            u = self.cpuct * child.P * sum_n_quarter / ((1 + child.N) ** 0.5)
            score = child.Q + u
            if score > best_score:
                best_score, best_act, best_child = score, a, child
        
        # Thực hiện hành động tốt nhất
        next_state = env.step(best_act)
        
        # Kiểm tra nếu game kết thúc sau khi thực hiện hành động
        if next_state.done:
            return path + [(best_act, best_child)], best_child, env, depth + 1
            
        # Tiếp tục selection đệ quy
        return self._select(env, best_child, path + [(best_act, best_child)], depth + 1)
    
    def _expand(self, env, node):
        """
        2. Expansion: Mở rộng node lá bằng cách tạo các node con
        """
        if node.children:
            return False
            
        obs, mask = self._encode_env(env)
        
        was_training = self.net.training
        self.net.eval()
        
        try:
            with torch.no_grad():
                obs = obs.to(self.device, non_blocking=self.non_blocking)
                P, v = self.net(obs.unsqueeze(0))
                
            P = P.squeeze(0).cpu().numpy() * mask
            if P.sum() > 0:
                P = P / P.sum()
            else:
                P = mask.astype(np.float32)
                P = P / P.sum()
            
            expanded = False
            for a, prob in enumerate(P):
                if mask[a]:
                    node.children[a] = MCTSNode(prior=prob)
                    expanded = True
                    
            return expanded
        finally:
            if was_training:
                self.net.train()
    
    def _simulate(self, env, node, expanded, depth):
        """
        3. Simulation: Mô phỏng đến khi game kết thúc hoặc đạt đến độ sâu tối đa
        """
        if self._game_is_over(env):
            return self._get_reward(env)
            
        if depth >= self.max_depth:
            obs, _ = self._encode_env(env)
            was_training = self.net.training
            self.net.eval()
            try:
                with torch.no_grad():
                    obs = obs.to(self.device, non_blocking=self.non_blocking)
                    _, v = self.net(obs.unsqueeze(0))
                return v.item()
            finally:
                if was_training:
                    self.net.train()
                    
        if not expanded:
            obs, _ = self._encode_env(env)
            was_training = self.net.training
            self.net.eval()
            try:
                with torch.no_grad():
                    obs = obs.to(self.device, non_blocking=self.non_blocking)
                    _, v = self.net(obs.unsqueeze(0))
                return v.item()
            finally:
                if was_training:
                    self.net.train()
        
        return self._get_reward(env)
    
    def _backup(self, path, v):
        """
        4. Backup: Cập nhật statistics cho các node trên đường đi
        """
        # Cập nhật statistics cho mỗi node trong path
        for action, node in reversed(path):
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
    
    def _game_is_over(self, env):
        """Kiểm tra xem game đã kết thúc chưa"""
        # Kiểm tra nếu có người chơi nào đã đạt điểm chiến thắng
        for player_idx in range(env.num_agents):
            if env.players[player_idx]['score'] >= 15:  # 15 là điểm chiến thắng trong Splendor
                return True
        return False
    
    def _get_reward(self, env):
        """Tính toán reward dựa trên điểm số của người chơi và đối thủ"""
        current_player_score = env.players[env.current_player_index]['score']
        opponent_scores = [env.players[i]['score'] for i in range(env.num_agents) if i != env.current_player_index]
        max_opponent_score = max(opponent_scores)
        
        # Tính giá trị dựa trên cả điểm của bản thân và điểm của đối thủ
        reward = 0.7 * current_player_score - 0.3 * max_opponent_score
        
        if self.debug_mode:
            self.logger.debug(f"Reward calculation: current_score={current_player_score}, max_opponent_score={max_opponent_score}, reward={reward}")
            
        return reward

    def _encode_env(self, env : SplendorLightZeroEnv):
        state = env.observe(env.current_player_index)
        # Update observation preprocessing to match train_dueling approach
        state_array = np.concatenate([
            state['tier1'].flatten(), 
            state['tier2'].flatten(), 
            state['tier3'].flatten(), 
            state['tokens'], 
            state['current_player'], 
            state['nobles'].flatten()
        ])
        # Đưa tensor vào cùng thiết bị với model
        x = torch.tensor(state_array, dtype=torch.float32).to(self.device)
        mask = env.get_action_mask()
        return x, mask 