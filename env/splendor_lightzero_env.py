import logging
import os
import copy
import numpy as np
import pandas as pd
from gymnasium import spaces
from ding.envs.env.base_env import BaseEnvTimestep, BaseEnv
from ding.utils.registry_factory import ENV_REGISTRY
import itertools as it

# Constants
GOLD = 5
NOT_GOLD = 7
NOBLES = 5
NOBLEMAN_VALUE = 3
WINNING_SCORE = 15
MAXIMUM_TOKENS = 10
MAXIMUM_RESERVATIONS = 3

@ENV_REGISTRY.register('splendor_lightzero')
class SplendorLightZeroEnv(BaseEnv):
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.max_episode_steps = cfg.get('max_episode_steps', 500)
        self.battle_mode = cfg.get('battle_mode', 'self_play_mode')
        self.replay_path = cfg.get('replay_path', None)
        self.prob_random = cfg.get('prob_random_agent', 0.0)
        self.prob_expert = cfg.get('prob_expert_agent', 0.0)

        # Load CSV data
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cards_csv = os.path.join(base_dir, "cards.csv")
        self.nobles_csv = os.path.join(base_dir, "nobles.csv")
        self.primary_cards = pd.read_csv(self.cards_csv)
        self.primary_nobles = pd.read_csv(self.nobles_csv)

        self.num_agents = 4
        self.colors = ['green', 'white', 'blue', 'black', 'red']

        # Define pick_tokens actions
        self.pick_tokens = []
        for k in range(1, 4):
            for combo in it.combinations(self.colors, k):
                self.pick_tokens.append({color: 1 if color in combo else 0 for color in self.colors})
        for color in self.colors:
            self.pick_tokens.append({c: 2 if c == color else 0 for c in self.colors})

        # Action space
        self.output_nodes = len(self.pick_tokens) + 12 + 12 + MAXIMUM_RESERVATIONS + len(self.pick_tokens) + 1
        self._action_space = spaces.Discrete(self.output_nodes)

        # Observation space
        card_feature_dim = self.primary_cards.shape[1]
        self.card_feature_dim = self.primary_cards.shape[1]
        self._observation_space = spaces.Dict({
            "tier1": spaces.Box(low=0, high=7, shape=(4, card_feature_dim), dtype=np.int32),
            "tier2": spaces.Box(low=0, high=7, shape=(4, card_feature_dim), dtype=np.int32),
            "tier3": spaces.Box(low=0, high=7, shape=(4, card_feature_dim), dtype=np.int32),
            "tokens": spaces.Box(low=0, high=MAXIMUM_TOKENS, shape=(6,), dtype=np.int32),
            "current_player": spaces.Box(low=0, high=40, shape=(6 + 5 + card_feature_dim * MAXIMUM_RESERVATIONS,), dtype=np.int32),
            "nobles": spaces.Box(low=0, high=4, shape=(NOBLES, 5), dtype=np.int32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.output_nodes,), dtype=np.int8)
        })
        self._reward_space = spaces.Box(low=0, high=100, shape=(self.num_agents,), dtype=np.float32)

        # Add player step counters
        self.player_steps = [0] * 4

        self.reset()

    def reset(self, start_player_index=0, init_state=None):
        self.episode = 0
        self.current_player_index = start_player_index
        self.frames = [] if self.replay_path else None

        if init_state is None:
            self.place_tokens()
            self.set_cards()
            self.create_players()
        else:
            self.tokens = init_state['tokens']
            self.tier1 = init_state['tier1']
            self.tier2 = init_state['tier2']
            self.tier3 = init_state['tier3']
            self.nobles = init_state['nobles']
            self.players = init_state['players']
            
        self.last_players = copy.deepcopy(self.players)
        obs = self.observe(self.current_player_index)
        return {
            'observation': obs,
            'action_mask': self.get_action_mask(),
            'to_play': self.current_player_index if self.battle_mode == 'self_play_mode' else -1
        }

    def step(self, action):
        if self.battle_mode == 'self_play_mode':
            if np.random.random() < self.prob_random:
                action = self.random_action()
            elif np.random.random() < self.prob_expert:
                action = self.bot_action()
            return self._step(action)

        elif self.battle_mode == 'play_with_bot_mode':
            timestep = self._step(action)
            if not timestep.done:
                bot_action = self.bot_action()
                return self._step(bot_action)
            return timestep

        elif self.battle_mode == 'eval_mode':
            timestep = self._step(action)
            if self.frames is not None:
                self.frames.append(self.render())
            if timestep.done and self.replay_path:
                self.save_replay()
            return timestep
        
    def get_action_type(self, action):
        if action < len(self.pick_tokens):  # Pick tokens
            current_action = "pick token"
        elif action < len(self.pick_tokens) + 12:  # Buy card
            current_action = "buy card"
        elif action < len(self.pick_tokens) + 24:  # Reserve card
            # print(f"Player {self.current_player_index} reserves card: {action}")
            current_action = "reserve card"
        elif action < len(self.pick_tokens) + 24 + MAXIMUM_RESERVATIONS:  # Buy reserved
            current_action = "buy reserved"
                
        elif action < len(self.pick_tokens) + 24 + MAXIMUM_RESERVATIONS + len(self.pick_tokens):  # Return tokens
            current_action = "return token"
        else:
            current_action = "skip"
        return current_action
    
    def _step(self, action):
        action_mask = self.get_action_mask()
        if not action_mask[action]:
            action = self.random_action()
            
        current_action = "skip"
        # type of action
        
        prev_score = self.players[self.current_player_index]['score']
        prev_card = sum(self.players[self.current_player_index]['cards'].values())
        self._execute_action(action)
        self.check_nobles()
        
        # Tính toán phần thưởng dựa trên điểm số và thẻ
        score_increase = self.players[self.current_player_index]['score'] - prev_score
        card_increase = sum(self.players[self.current_player_index]["cards"].values()) - prev_card
        
        # Phần thưởng cơ bản
        reward = card_increase + (score_increase ** 2)  # Tăng trọng số cho điểm số
        
        # Phạt nếu không có tiến triển
        if reward == 0: 
            reward = -0.8  # Tăng mức phạt để khuyến khích hành động có hiệu quả
            
        # Check if any player has reached winning score
        game_ended = any(player['score'] >= WINNING_SCORE for player in self.players)
        
        # Game ends if max steps reached or if we've completed the round after someone reached winning score
        done = self.episode >= self.max_episode_steps or (game_ended and self.current_player_index == 0)
        
        # Kiểm tra tổng token sau hành động
        total_tokens = sum(self.players[self.current_player_index]['tokens'].values())
        
        # Chỉ tăng step và chuyển lượt nếu không phải return token và total_tokens <= MAXIMUM_TOKENS
        is_return_token = len(self.pick_tokens) + 24 + MAXIMUM_RESERVATIONS <= action and action < len(self.pick_tokens) + 24 + MAXIMUM_RESERVATIONS + len(self.pick_tokens)
        
        if not is_return_token and (total_tokens <= MAXIMUM_TOKENS or not is_return_token):
            self.episode += 1
            self.player_steps[self.current_player_index] += 1
            self.current_player_index = (self.current_player_index + 1) % self.num_agents
        
        obs = self.observe(self.current_player_index)
        info = {
            'eval_episode_return': self.players[self.current_player_index]['score'], 
            'episode': self.episode, 
            'player': self.current_player_index,
            'player_steps': self.player_steps
        } if done else {
            'player': self.current_player_index,
            'player_steps': self.player_steps
        }
        
        self.last_players = copy.deepcopy(self.players)
        
        return BaseEnvTimestep(obs, reward, done, info)

    def _execute_action(self, action):
        player = self.players[self.current_player_index]
        
        if action < len(self.pick_tokens):  # Pick tokens
            # print(f"Player {self.current_player_index} picks tokens: {self.pick_tokens[action]}")
            for color in self.colors:
                player['tokens'][color] += self.pick_tokens[action][color]
                self.tokens[color] -= self.pick_tokens[action][color]
                
        elif action < len(self.pick_tokens) + 12:  # Buy card
            # print(f"Player {self.current_player_index} buys card: {action}")
            card_idx = (action - len(self.pick_tokens)) % 4
            card = self.tier1.iloc[card_idx] if action < len(self.pick_tokens) + 4 else self.tier2.iloc[card_idx] if action < len(self.pick_tokens) + 2 * 4 else self.tier3.iloc[card_idx]
            self.buy(card)
            
        elif action < len(self.pick_tokens) + 24:  # Reserve card
            # print(f"Player {self.current_player_index} reserves card: {action}")
            
            card_idx = (action - len(self.pick_tokens) - 12) % 4
            card = self.tier1.iloc[card_idx] if action < len(self.pick_tokens) + 12 + 4 else self.tier2.iloc[card_idx] if action < len(self.pick_tokens) + 12 + 2 * 4 else self.tier3.iloc[card_idx]
            self.reserve(card)
            
        elif action < len(self.pick_tokens) + 24 + MAXIMUM_RESERVATIONS:  # Buy reserved
            # print(f"Player {self.current_player_index} buys reserved card: {action}")
            idx = action - len(self.pick_tokens) - 24
            if idx < len(player['reservations']):
                self.buy(player['reservations'].pop(idx))
                
        elif action < len(self.pick_tokens) + 24 + MAXIMUM_RESERVATIONS + len(self.pick_tokens):  # Return tokens
            # print(f"Player {self.current_player_index} returns tokens: {action}")
            tokens = self.pick_tokens[action - len(self.pick_tokens) - 24 - MAXIMUM_RESERVATIONS]
            self.do_return_tokens(tokens)
            
        else:  # Skip
            pass

    def observe(self, player_index):
        state = self.return_state(player_index)
        state['action_mask'] = self.get_action_mask(player_index)
        state['to_play'] = self.current_player_index
        return state

    def return_state(self, player_index):
        player = self.players[player_index]
        tokens = np.array([self.tokens[c] for c in self.colors + ['gold']], dtype=np.int32)
        
        # Chuẩn hóa tiers
        tier1_flat = np.zeros((4 * self.card_feature_dim,), dtype=np.int32)
        tier2_flat = np.zeros((4 * self.card_feature_dim,), dtype=np.int32)
        tier3_flat = np.zeros((4 * self.card_feature_dim,), dtype=np.int32)
        if not self.tier1.empty:
            tier1_data = self.tier1.head(4).to_numpy().flatten()
            tier1_flat[:len(tier1_data)] = tier1_data
        if not self.tier2.empty:
            tier2_data = self.tier2.head(4).to_numpy().flatten()
            tier2_flat[:len(tier2_data)] = tier2_data
        if not self.tier3.empty:
            tier3_data = self.tier3.head(4).to_numpy().flatten()
            tier3_flat[:len(tier3_data)] = tier3_data
        
        # Chuẩn hóa reservations
        reservations_flat = np.zeros((MAXIMUM_RESERVATIONS * self.card_feature_dim,), dtype=np.int32)
        if player['reservations']:
            reservations_data = np.array([card.to_numpy() if isinstance(card, pd.Series) else card for card in player['reservations']], dtype=np.int32).flatten()
            reservations_flat[:len(reservations_data)] = reservations_data[:MAXIMUM_RESERVATIONS * self.card_feature_dim]
        
        player_data = np.concatenate([
            np.array([player['tokens'][c] for c in self.colors + ['gold']], dtype=np.int32),
            np.array([player['cards'][c] for c in self.colors], dtype=np.int32),
            reservations_flat
        ])
        
        # Chuẩn hóa nobles
        nobles_flat = np.zeros((NOBLES * 5,), dtype=np.int32)
        if not self.nobles.empty:
            nobles_data = self.nobles.to_numpy().flatten()
            nobles_flat[:len(nobles_data)] = nobles_data[:NOBLES * 5]
        
        return {
            "tier1": tier1_flat.reshape(4, self.card_feature_dim),
            "tier2": tier2_flat.reshape(4, self.card_feature_dim),
            "tier3": tier3_flat.reshape(4, self.card_feature_dim),
            "tokens": tokens,
            "current_player": player_data,
            "nobles": nobles_flat
        }

    def get_action_mask(self, player_index=None):
        if player_index is None:
            player_index = self.current_player_index
        mask = np.zeros(self.output_nodes, dtype=np.int8)
        player = self.players[player_index]
        total_tokens = sum(player['tokens'].values())

        # Nếu tổng token vượt quá MAXIMUM_TOKENS (10), chỉ cho phép trả token hoặc skip
        if total_tokens > MAXIMUM_TOKENS:
            # Return tokens
            for i, tokens in enumerate(self.pick_tokens):
                if all(player['tokens'][c] >= tokens[c] for c in self.colors):
                    mask[len(self.pick_tokens) + 24 + MAXIMUM_RESERVATIONS + i] = 1
            
            # Skip luôn khả dụng nếu không có hành động trả token nào
            if sum(mask) == 0:
                mask[-1] = 1
                
        else:
            # Pick tokens (không giới hạn tổng token khi lấy)
            for i, tokens in enumerate(self.pick_tokens):
                if all(self.tokens[c] >= tokens[c] for c in self.colors):  # Chỉ kiểm tra đủ token trên bàn
                    mask[i] = 1

            # Buy cards
            for i in range(12):
                tier = i // 4
                card_idx = i % 4
                if card_idx < len([self.tier1, self.tier2, self.tier3][tier]):
                    card = [self.tier1, self.tier2, self.tier3][tier].iloc[card_idx]
                    if self.can_afford(card):
                        mask[len(self.pick_tokens) + i] = 1

            # Reserve cards
            if len(player['reservations']) < MAXIMUM_RESERVATIONS:
                for i in range(12):
                    tier = i // 4
                    card_idx = i % 4
                    if card_idx < len([self.tier1, self.tier2, self.tier3][tier]):
                        mask[len(self.pick_tokens) + 12 + i] = 1

            # Buy reserved
            for i, card in enumerate(player['reservations'][:MAXIMUM_RESERVATIONS]):
                if self.can_afford(card):
                    mask[len(self.pick_tokens) + 24 + i] = 1

            # Skip luôn khả dụng nếu không có hành động nào khác
            if sum(mask) == 0:
                mask[-1] = 1

        return mask

    def place_tokens(self):
        self.tokens = {c: NOT_GOLD for c in self.colors}
        self.tokens['gold'] = GOLD

    def set_cards(self):
        shuffled_cards = self.primary_cards.sample(frac=1).reset_index(drop=True)
        self.tier1 = shuffled_cards[shuffled_cards['tier'] == 1].reset_index(drop=True)
        self.tier2 = shuffled_cards[shuffled_cards['tier'] == 2].reset_index(drop=True)
        self.tier3 = shuffled_cards[shuffled_cards['tier'] == 3].reset_index(drop=True)
        self.nobles = self.primary_nobles.sample(frac=1).reset_index(drop=True).head(NOBLES)

    def create_players(self):
        primary_player = {
            'score': 0,
            'tokens': {c: 0 for c in self.colors + ['gold']},
            'cards': {c: 0 for c in self.colors},
            'reservations': []
        }
        self.players = [copy.deepcopy(primary_player) for _ in range(self.num_agents)]

    def can_afford(self, card):
        player = self.players[self.current_player_index]
        gold_needed = 0
        for color in self.colors:
            cost = max(0, card[color] - player['cards'][color])
            gold_needed += max(0, cost - player['tokens'][color])
        return gold_needed <= player['tokens']['gold']

    def buy(self, card):
        player = self.players[self.current_player_index]
        for color in self.colors:
            cost = max(0, card[color] - player['cards'][color])
            if cost > player['tokens'][color]:
                gold_cost = cost - player['tokens'][color]
                player['tokens'][color] = 0
                player['tokens']['gold'] -= gold_cost
                self.tokens['gold'] += gold_cost
            else:
                player['tokens'][color] -= cost
            self.tokens[color] += cost
            
        player['cards'][self.colors[int(card['color']) - 1]] += 1
        player['score'] += int(card['value'])
        
        if card.name in self.tier1.index:
            self.tier1 = self.tier1.drop(card.name).reset_index(drop=True)
        elif card.name in self.tier2.index:
            self.tier2 = self.tier2.drop(card.name).reset_index(drop=True)
        elif card.name in self.tier3.index:
            self.tier3 = self.tier3.drop(card.name).reset_index(drop=True)

    def reserve(self, card):
        player = self.players[self.current_player_index]
        player['reservations'].append(card)
        if self.tokens['gold'] > 0:
            player['tokens']['gold'] += 1
            self.tokens['gold'] -= 1
        if card.name in self.tier1.index:
            self.tier1 = self.tier1.drop(card.name).reset_index(drop=True)
        elif card.name in self.tier2.index:
            self.tier2 = self.tier2.drop(card.name).reset_index(drop=True)
        elif card.name in self.tier3.index:
            self.tier3 = self.tier3.drop(card.name).reset_index(drop=True)

    def do_return_tokens(self, tokens):
        player = self.players[self.current_player_index]
        for color in self.colors:
            player['tokens'][color] -= tokens[color]
            self.tokens[color] += tokens[color]

    def check_nobles(self):
        player = self.players[self.current_player_index]
        for i, noble in self.nobles.iterrows():
            if all(player['cards'][c] >= noble[c] for c in self.colors):
                player['score'] += NOBLEMAN_VALUE
                self.nobles = self.nobles.drop(i).reset_index(drop=True)
                break

    def random_action(self):
        return np.random.choice(np.where(self.get_action_mask())[0])

    def bot_action(self):
        legal = np.where(self.get_action_mask())[0]
        best_action = legal[0]
        best_score = -float('inf')
        for action in legal:
            temp_env = copy.deepcopy(self)
            temp_env._execute_action(action)
            score = temp_env.players[self.current_player_index]['score']
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def render(self):
        state = self.return_state(self.current_player_index)
        return f"Tokens: {state['tokens']}\nTiers: {state['tier1']}, {state['tier2']}, {state['tier3']}\nPlayer: {self.players[self.current_player_index]}\nNobles: {state['nobles']}"

    def save_replay(self):
        if not os.path.exists(self.replay_path): # type: ignore
            os.makedirs(self.replay_path) # type: ignore
        path = os.path.join(self.replay_path, f'replay_{len(os.listdir(self.replay_path))}.txt') # type: ignore
        with open(path, 'w') as f:
            f.write('\n'.join(self.frames)) # type: ignore
            
    
        
    @property
    def legal_actions(self):
        return np.where(self.get_action_mask())[0]

    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        pass

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    def __repr__(self):
        return "LightZero Splendor Env"