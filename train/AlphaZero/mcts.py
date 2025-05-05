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
        self.max_depth = config.get('max_depth', 100)
        # Lưu thiết bị của model
        self.device = next(net.parameters()).device
        # Khởi tạo logger
        self.logger = logging.getLogger("AlphaZero.MCTS")
        self.debug_mode = config.get('debug_mode', False)

    def run(self, env : SplendorLightZeroEnv):
        # self.logger.debug(f"Starting MCTS search for player {env.current_player_index}")
        root = MCTSNode(prior=1.0)
        obs, mask = self._encode_env(env)
        
        legal_actions_count = np.sum(mask)
        self.logger.debug(f"Legal actions: {legal_actions_count}/{len(mask)}")
        
        # Kiểm tra nếu không có hành động hợp lệ
        if legal_actions_count == 0:
            self.logger.warning("No legal actions available!")
            # Trả về phân phối đều cho tất cả hành động
            pi = np.ones_like(mask) / len(mask)
            return pi
            
        # Lưu trạng thái train hiện tại
        was_training = self.net.training
        self.net.eval()  # Chuyển sang chế độ eval để tránh lỗi BatchNorm với batch size 1
        try:
            # Dự đoán prior probabilities từ mạng
            with torch.no_grad():
                P, v = self.net(obs.unsqueeze(0))
                self.logger.debug(f"Initial value prediction: {v.item():.4f}")
                
            P = P.squeeze(0).cpu().numpy() * mask
            if P.sum() > 0:
                P = P / P.sum()
            else:
                # Nếu tất cả prior probability bằng 0, sử dụng phân phối đều cho các hành động hợp lệ
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
                # Kiểm tra lỗi số học float
                if abs(pi.sum() - 1.0) > 1e-9:
                    # self.logger.warning(f"Pi sum is {pi.sum()}, renormalizing")
                    pi = pi / pi.sum()
            else:
                # Nếu không có lần thăm nào, sử dụng phân phối đều trên các hành động hợp lệ
                self.logger.warning("No visits recorded, using uniform distribution")
                pi = mask.astype(np.float32)
                pi = pi / pi.sum()
            
            # Kiểm tra cuối cùng để đảm bảo tổng bằng 1
            assert np.isclose(pi.sum(), 1.0), f"Pi sum is {pi.sum()}, not 1.0"
                
            # Log các hành động hàng đầu dựa trên số lần thăm
            # if self.debug_mode:
            #     top_actions = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            #     self.logger.debug("Top actions (action, visit count, probability):")
            #     for a, count in top_actions:
            #         self.logger.debug(f"  Action {a}: visits={count}, prob={pi[a]:.4f}")
            
            # self.logger.debug(f"MCTS search completed with {self.num_sim} simulations")
            return pi
        finally:
            # Khôi phục lại trạng thái train ban đầu
            if was_training:
                self.net.train()
    
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
        total_N = sum(child.N for child in node.children.values())
        
        # Tính UCB cho mỗi hành động
        for a, child in node.children.items():
            u = self.cpuct * child.P * math.sqrt(total_N) / (1 + child.N)
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
        # Nếu node đã có con, không cần mở rộng
        if node.children:
            return False
            
        # Lấy observation và action mask
        obs, mask = self._encode_env(env)
        
        # Đảm bảo mạng ở chế độ eval
        was_training = self.net.training
        self.net.eval()
        
        try:
            # Dự đoán prior probabilities từ mạng
            with torch.no_grad():
                P, v = self.net(obs.unsqueeze(0))
                
            # Áp dụng action mask và chuẩn hóa
            P = P.squeeze(0).cpu().numpy() * mask
            if P.sum() > 0:
                P = P / P.sum()
            else:
                P = mask.astype(np.float32)
                P = P / P.sum()
            
            # Tạo các node con
            expanded = False
            for a, prob in enumerate(P):
                if mask[a]:
                    node.children[a] = MCTSNode(prior=prob)
                    expanded = True
                    
            return expanded
        finally:
            # Khôi phục lại trạng thái train ban đầu
            if was_training:
                self.net.train()
    
    def _simulate(self, env, node, expanded, depth):
        """
        3. Simulation: Mô phỏng đến khi game kết thúc hoặc đạt đến độ sâu tối đa
        """
        # Nếu game đã kết thúc, trả về reward
        if self._game_is_over(env):
            return self._get_reward(env)
            
        # Nếu đạt đến độ sâu tối đa, dự đoán giá trị của trạng thái hiện tại
        if depth >= self.max_depth:
            obs, _ = self._encode_env(env)
            was_training = self.net.training
            self.net.eval()
            try:
                with torch.no_grad():
                    _, v = self.net(obs.unsqueeze(0))
                return v.item()
            finally:
                if was_training:
                    self.net.train()
                    
        # Nếu node không được mở rộng (không có node con), chỉ đánh giá giá trị hiện tại
        if not expanded:
            obs, _ = self._encode_env(env)
            was_training = self.net.training
            self.net.eval()
            try:
                with torch.no_grad():
                    _, v = self.net(obs.unsqueeze(0))
                return v.item()
            finally:
                if was_training:
                    self.net.train()
        
        # Nếu có con, ta chọn một node con để mô phỏng tiếp
        # Ở Alpha Zero, không cần rollout ngẫu nhiên tới cuối game
        # mà sử dụng giá trị dự đoán từ mạng neural
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