import os
import copy
import json
import logging
import itertools as it
from collections import defaultdict
from datetime import datetime
import sys
from typing import List

import numpy as np
import pandas as pd
from gymnasium import spaces

from ding.envs.env.base_env import BaseEnvTimestep,BaseEnv
from ding.utils.registry_factory import ENV_REGISTRY
import lzero 
from pettingzoo.utils.agent_selector import agent_selector
import logging
# --- Các hằng số của Splendor ---
GOLD = 5
NOT_GOLD = 7
NOBLES = 5
NOBLEMAN_VALUE = 3
WINNING_SCORE = 15
MAXIMUM_TOKENS = 10
MAXIMUM_RESERVATIONS = 3

@ENV_REGISTRY.register('splendor')
class SplendorEnv(BaseEnv):
    def __init__(self, cfg={}):   
        self.cfg = cfg
        self.max_episode_steps = 1000
          
        # Đường dẫn đến file CSV (mặc định trong cùng thư mục với file này)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cards_csv = os.path.join(base_dir, "cards.csv")
        self.nobles_csv = os.path.join(base_dir, "nobles.csv")
        
        # Load cấu hình deck và nobles từ CSV
        self.primary_cards = pd.read_csv(self.cards_csv)
        self.primary_nobles = pd.read_csv(self.nobles_csv)
        # print(self.primary_cards)
        # print(self.primary_nobles)
        # Đảm bảo rằng các cột trong CSV được định dạng đúng
        
        self.agents = ['Player_1', 'Player_2', 'Player_3', 'Player_4']
        self.num_agents = len(self.agents)
        self.possible_agents = self.agents[:]
        self.has_reset = False

        self.colors = ['green', 'white', 'blue', 'black', 'red']
        
        self.current_player_index = 0
        
        self.pick_tokens = []
        # Cấu hình các cách lấy token (sử dụng các chỉ số từ 1 đến 3 tương ứng với 5 màu)
        def C(n, k):
            for combo in it.combinations(range(n), k):
                arr = [0] * 5
                for i in combo:
                    arr[i] = 1
                token_dict = {color: count for color, count in zip(self.colors, arr)}
                self.pick_tokens.append(token_dict)

        for i in range(1, 4):
            C(5, i)

        for i in range(5):
            arr = [0] * 5
            arr[i] = 2
            token_dict = {color: count for color, count in zip(self.colors, arr)}
            self.pick_tokens.append(token_dict)
            
        # Tính tổng số output nodes (dùng cho mã hóa hành động)
        self.output_nodes = sum([
            len(self.pick_tokens), # Lấy token
            3 * 4 * 2,   # Mua và dự trữ thẻ trên bàn (3 tầng, 4 thẻ mỗi tầng, 2 loại hành động)
            3,           # Mua thẻ đã được dự trữ
            len(self.pick_tokens), # Trả token
            1            # Empty move
        ])

        # Không gian hành động: rời rạc từ 0 đến output_nodes - 1
        self._action_space = self._convert_to_dict(
            [spaces.Discrete(self.output_nodes)] for _ in range(self.num_agents)
        )
        
        
        card_feature_dim = self.primary_cards.shape[1]
        
        '''
        gems_on_table: 6 phần tử (1 vector 6 chiều).
        cards_on_table: 3 tầng, 4 thẻ x 8 thông số (điểm, màu thẻ, 5 requirement). 3 x 4 x 8 = 84 phần tử
        player_gems: 6 phần tử.
        player_cards: 5 phần tử.
        player_points: 1 phần tử.
        nobles: 5 quý tộc x 5 yêu cầu = 25 phần tử.
        opponents:
            Gems: 3 đối thủ x 6 loại đá = 18 phần tử.
            Cards: 3 đối thủ x 5 loại thẻ = 15 phần tử.
            Points: 3 đối thủ x 1 = 3 phần tử.
            Tổng: 36 phần tử.
        turn: 1 phần tử.
        '''
        self._observation_space = self._convert_to_dict(
            [
                spaces.Dict(
                    {
                        "tier1": spaces.Box(low=0, high=7, shape=(4, card_feature_dim), dtype=np.float32),
                        "tier2": spaces.Box(low=0, high=7, shape=(4, card_feature_dim), dtype=np.float32),
                        "tier3": spaces.Box(low=0, high=7, shape=(4, card_feature_dim), dtype=np.float32),
                        "tokens": spaces.Box(low=0, high=MAXIMUM_TOKENS, shape=(6,), dtype=np.int32),
                        "current_player": spaces.Box(
                            low=0, 
                            high=np.array(
                                [WINNING_SCORE] +              # score: 15
                                [MAXIMUM_TOKENS]*6 +           # tokens: 10 mỗi loại
                                [40]*5 +                       # cards đã mua có 5 màu: tối đa 40 mỗi loại
                                [7] * card_feature_dim * 3         # số thẻ reservations: max 3, mỗi thẻ có 8 thông số -> 24
                            ), 
                            shape=(36,),  # Giảm từ 14 xuống 13 (bỏ nobles)
                            dtype=np.int32
                        ),
                        "nobles": spaces.Box(low=0, high=4, shape=(NOBLES, 5), dtype=np.int32),
                        "action_mask": spaces.Box(low=0, high=1, shape=(self.output_nodes,), dtype=np.int8)
                    }
                )
            ] for _ in range(self.num_agents)
        )
        self._agent_selector = agent_selector(self.agents)
        self._reward_space = spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)

        # Khởi tạo trạng thái game ban đầu
        self.reset()

    def place_tokens(self):
        self.tokens = {
            'green': NOT_GOLD,
            'white': NOT_GOLD,
            'blue': NOT_GOLD,
            'black': NOT_GOLD,
            'red': NOT_GOLD,
            'gold': GOLD
        }

    def set_cards(self):
        # Trộn các lá bài và nobles
        shuffled_cards = self.primary_cards.sample(frac=1).reset_index(drop=True)
        shuffled_nobles = self.primary_nobles.sample(frac=1).reset_index(drop=True)
        # Phân loại lá bài theo tier (giả sử file CSV có cột 'tier')
        self.tier1 = shuffled_cards[shuffled_cards['tier'] == 1].reset_index(drop=True)
        self.tier2 = shuffled_cards[shuffled_cards['tier'] == 2].reset_index(drop=True)
        self.tier3 = shuffled_cards[shuffled_cards['tier'] == 3].reset_index(drop=True)
        # Chọn các nobles (chỉ lấy NOBLES)
        self.nobles = shuffled_nobles.tail(NOBLES).reset_index(drop=True)
        
        # print(self.tier1)
        # print(self.tier2)
        # print(self.tier3)
        # print(self.nobles)

    def create_players(self):
        # Khởi tạo trạng thái cho từng người chơi
        primary_player = {
            'score': 0,
            'tokens': {'green': 0, 'white': 0, 'blue': 0, 'black': 0, 'red': 0, 'gold': 0},
            'cards': {'green': 0, 'white': 0, 'blue': 0, 'black': 0, 'red': 0},
            'nobles': 0,
            'reservations': []
        }
        self.players = [copy.deepcopy(primary_player) for _ in range(self.num_agents)]

        
    def reset(self):
        # reset the environment...
        self.has_reset = False
        
        self.episode = 0
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        
        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        
        self.return_tokens = False
        self.set_cards()
        self.place_tokens()
        self.create_players()
        # print(self.tokens)
        self.agents = self.possible_agents[:]
        self.current_player_index = 0
    
        agent = self.agent_selection
        current_index = self.agents.index(agent)
        self.current_player_index = current_index
        obs = self.observe(agent)
        action_mask = self.get_action_mask()
        
        lightzero_obs_dict = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}
        return lightzero_obs_dict

    def step(self, action):
        """ 
        Hàm step nhận một hành động (action) và cập nhật trạng thái game.
        Nếu score >= WINNING_SCORE thì game kết thúc.
        Sau đó, chuyển lượt sang người chơi tiếp theo.
        """
        self.episode += 1
    
        self.last_obs = self.observe(self.agent_selection)
        # # print(f"Last obs: {self.last_obs}")
        # print(f"Current player: {self.agent_selection}") 
        # # print(f"Current player index: {self.current_player}")
        # print(f"Action: {action}")
        # print(f"Legal actions: {self.legal_actions}")
        # print(f"tokens: {self.tokens}")
        # print(f"current player tokens: {self.players[self.current_player]['tokens']}")
        # print(f"Action: {action}")
        if(action not in self.legal_actions):
            logging.warning(
                f"Illegal action: {action}. Legal actions: {self.legal_actions}. "
                "Choosing a random action from legal actions."
            )
            action = np.random.choice(self.legal_actions)

        if(action < len(self.pick_tokens)):
            # Pick tokens
            print("pick tokens")
            tokens = self.pick_tokens[action]
            for color in tokens:
                self.players[self.current_player]['tokens'][color] += tokens[color]
                self.tokens[color] -= tokens[color]
                
        elif(action < len(self.pick_tokens) + 3 * 4 ):
            # Buy cards
            print("buy cards")
            card_idx = (action - len(self.pick_tokens)) % 4
            card = self.tier1.iloc[card_idx] if action < len(self.pick_tokens) + 4 else self.tier2.iloc[card_idx] if action < len(self.pick_tokens) + 2 * 4 else self.tier3.iloc[card_idx]
            print("Card:", card.to_dict())
            self.buy(card)
            
        elif(action < len(self.pick_tokens) + 3 * 4 * 2):
            # reserved card
            print("reserved card")
            card_idx = (action - (len(self.pick_tokens) - 12 + 4)) % 4
            card = self.tier1.iloc[card_idx] if action < len(self.pick_tokens) + 4 else self.tier2.iloc[card_idx] if action < len(self.pick_tokens) + 2 * 4 else self.tier3.iloc[card_idx]
            self.remove_card(card)
            self.reserve(card)
            
        elif(action < len(self.pick_tokens) + 3 * 4 * 2 + 3):
            # buy reserved card
            print("buy reserved card")
            card = self.players[self.current_player]['reservations'][action - len(self.pick_tokens) - 3 * 4 * 2]
            self.buy(card)
            
        elif(action < len(self.pick_tokens) + 3 * 4 * 2 + 3 + len(self.pick_tokens)):
            # return tokens
            print("return tokens")
            returning_tokens = self.pick_tokens[action - len(self.pick_tokens) - 3 * 4 * 2 - 3]
            self.do_return_tokens(returning_tokens)
        else:
            # skip 
            print("skip")
        
        # print("current player", self.current_player)
        # print("current player score", self.players[self.current_player]['score'])
        # print("agent: ", self.agent_selection)
        self.rewards[self.agent_selection] = self.players[self.current_player]["score"]
        
        self.check_nobles()
        done = False
        # if self.episode >= self.max_episode_steps:
        #     done = True
        if self.rewards[self.agent_selection] >= WINNING_SCORE:
            done = True
            
        prev_agent = self.agent_selection
        self.agent_selection = self._agent_selector.next()
        agent = self.agent_selection
        self.next_player_index = self.agents.index(agent)
        
        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)
        self.current_player_index = current_index
        
        obs = self.observe(self.agent_selection)
        lightzero_obs_dict = {'observation': obs, 'action_mask': self.legal_actions, 'to_play': -1}
        
        self._eval_episode_return = self.rewards[agent]
        if done:
            self.infos[agent] = self._eval_episode_return
        # print(agent)
        return BaseEnvTimestep(lightzero_obs_dict, self.rewards[prev_agent], done, self.infos[agent])

    def return_state(self):
        card_feature_dim = self.primary_cards.shape[1]  # Ví dụ: 8

        def get_tier_obs(tier_df):
            num_shown = min(4, len(tier_df))
            arr = tier_df.tail(num_shown).to_numpy(dtype=np.float32)
            padded = np.zeros((4, card_feature_dim), dtype=np.float32)
            if arr.shape[0] > 0:
                padded[-arr.shape[0]:] = arr
            return padded

        tier1_obs = get_tier_obs(self.tier1)
        tier2_obs = get_tier_obs(self.tier2)
        tier3_obs = get_tier_obs(self.tier3)

        tokens_arr = np.array([self.tokens[color] for color in ['green', 'white', 'blue', 'black', 'red', 'gold']], dtype=np.int32)

        # Thông tin người chơi hiện tại
        current_player = self.players[self.current_player]
        reservations_arr = np.zeros((MAXIMUM_RESERVATIONS * card_feature_dim), dtype=np.int32)  # 3 x 8 = 24
        num_reservations = min(len(current_player['reservations']), MAXIMUM_RESERVATIONS)
        if num_reservations > 0:
            reservations_arr[-num_reservations * card_feature_dim:] = np.concatenate(
                [card.to_numpy(dtype=np.int32) for card in current_player['reservations'][-num_reservations:]]
            )

        current_player_arr = np.concatenate(
            [
                [current_player['score']],
                [current_player['tokens'][color] for color in ['green', 'white', 'blue', 'black', 'red', 'gold']],
                [current_player['cards'][color] for color in ['green', 'white', 'blue', 'black', 'red']],
                reservations_arr
            ],
            dtype=np.int32
        )

        # Thông tin đối thủ (giữ nguyên)
        opponents_arr = np.zeros((self.num_agents-1, 4), dtype=np.int32)
        opponent_idx = 0
        for i, player in enumerate(self.players):
            if i != self.current_player:
                total_tokens = sum(player['tokens'].values())
                total_cards = sum(player['cards'].values())
                opponents_arr[opponent_idx] = [player['score'], total_tokens, total_cards, len(player['reservations'])]
                opponent_idx += 1

        # Quý tộc trên bàn (giữ nguyên)
        nobles_arr = np.zeros((NOBLES, 5), dtype=np.int32)
        if not self.nobles.empty:
            nobles_arr[:len(self.nobles)] = self.nobles[['green', 'white', 'blue', 'black', 'red']].to_numpy(dtype=np.int32)

        state = {
            "tier1": tier1_obs,
            "tier2": tier2_obs,
            "tier3": tier3_obs,
            "tokens": tokens_arr,
            "current_player": current_player_arr,  # Vector 36 phần tử
            "opponents": opponents_arr,
            "nobles": nobles_arr
        }
        return state
    
    def print_pick_token_colors(self):
        for action in range (len(self.pick_tokens)):
            tokens = self.pick_tokens[action]
            colors = [(color, count) for color, count in tokens.items() if count > 0]
            print(f"Các màu token tương ứng với action {action}:", colors)
    
    def flatten_observation(self, obs):
        flat_obs = np.concatenate([
            obs["tier1"].flatten(),
            obs["tier2"].flatten(),
            obs["tier3"].flatten(),
            obs["tokens"],
            obs["current_player"]["stats"],
            obs["current_player"]["reservations"].flatten(),
            obs["opponents"].flatten(),
            obs["nobles"].flatten()
        ])
        return flat_obs  # Shape: (175,)

    @property
    def legal_actions(self):
        # Ví dụ: tất cả các chỉ số từ 0 đến output_nodes - 1 đều hợp lệ
        valid_action = []
        # print("current player", self.current_player)
        # print("agent", self.agent_selection)
        
        for action in range(self.output_nodes):
            if(action < len(self.pick_tokens)):
                # Pick tokens
                if self.can_pick(self.pick_tokens[action]) and self.players[self.current_player]['tokens']:
                    valid_action.append(action)
                pass
            elif(action < len(self.pick_tokens) + 3 * 4 ):
                # Buy cards
                card_idx = (action - (len(self.pick_tokens) + 4)) % 4
                # print("card_idx", card_idx)
                card = self.tier1.iloc[card_idx] if action < len(self.pick_tokens) + 4 else self.tier2.iloc[card_idx] if action < len(self.pick_tokens) + 2 * 4 else self.tier3.iloc[card_idx]
                
                # Check if player has enough tokens to buy this card
                if self.can_afford(card):
                    valid_action.append(action)
                    # print("can buy card", action)
                pass
            elif(action < len(self.pick_tokens) + 3 * 4 * 2):
                # reserved card
                if len(self.players[self.current_player]['reservations']) < MAXIMUM_RESERVATIONS :
                    valid_action.append(action)
                pass
            elif(action < len(self.pick_tokens) + 3 * 4 * 2 + 3):
                # buy reserved card
                # Check if player has this card in reservations
                if(action - len(self.pick_tokens) - 3 * 4 * 2 + 1 > len(self.show_reservations()) or (self.show_reservations() == [])):
                    continue
                
                # Check if player can afford this card
                if self.can_afford(self.players[self.current_player]['reservations'][action - len(self.pick_tokens) - 3 * 4 * 2]):
                    valid_action.append(action)
                pass
            elif(action < len(self.pick_tokens) + 3 * 4 * 2 + 3 + len(self.pick_tokens)):
                # return tokens
                if self.can_return(self.pick_tokens[action - len(self.pick_tokens) - 3 * 4 * 2 - 3]):
                    valid_action.append(action)
                pass
            else:
                # skip 
                # valid_action.append(action)
                pass
            
        if(valid_action == []):
            valid_action.append(self.output_nodes - 1)
        return valid_action    
        
    def observe(self, agent_index):
        """
        Trả về quan sát cho người chơi có chỉ số agent_index dưới dạng dict với các khóa:
          - 'tier1', 'tier2', 'tier3', 'tokens', 'players' và 'action_mask'
        Phần 'action_mask' được khởi tạo .
        """
        obs = self.return_state()
        action_mask = np.zeros(self.output_nodes, dtype=np.int8)
        for action in self.legal_actions:
            action_mask[action] = 1
        obs["action_mask"] = action_mask
        return obs
    
    def card_to_colors(self, card):
        # Returns neccesary tokens for this card separated by comas
        return ','.join(str(int(card[c])) for c in self.colors)

    def show_reservations(self):
        reservations = self.players[self.current_player]['reservations']
        return [self.card_to_colors(reservations[i]) for i in range(len(reservations))]
    
    def can_pick(self, tokens: dict):
		# Unindexed amouns of certain tokens to pick
        values = np.array(list(tokens.values()))
        # print(values)
        if sum(values) + sum(self.players[self.current_player]['tokens'].values()) > MAXIMUM_TOKENS:
            return False
        # Player can not pick more tokens than there is on board 
        for color in tokens:
            if self.tokens[color] < tokens[color]:
                # print("sai lan 3", color, tokens[color], self.tokens[color])
                return False
            if(self.tokens[color] < 4 and tokens[color] == 2):
                # print("sai lan 4")
                return False
            
        return True

    def can_afford(self, card):
        # Current player assets
        tokens = self.players[self.current_player]['tokens']
        cards = self.players[self.current_player]['cards']
        # print("current player", self.agent_selection)
        # print("tokens", tokens)
        # print("cards of player", cards)
        # print("card", card)
        # Tokens needed to buy this card by current player
        token_diff = [tokens[i] + cards[i] - card[i] for i in self.colors]
        # print(token_diff)
        missing_tokens = abs(sum(i for i in token_diff if i < 0))

        # Check if player has enough tokens to buy this card
        if self.players[self.current_player]['tokens']['gold'] < missing_tokens:
            return False

        return True
    
    def can_return(self, returning_tokens):
        if sum(self.players[self.current_player]['tokens'].values()) < MAXIMUM_TOKENS:
            return False
        returning_amount = sum(returning_tokens.values())

        tokens = self.players[self.current_player]['tokens']
        current_amount = sum(tokens.values())

        if current_amount - returning_amount > 10:
            return False

        if any(tokens[i] < returning_tokens[i] for i in returning_tokens):
            return False

        return True
    
    def do_return_tokens(self, requested_tokens):
        for color in requested_tokens:
            self.players[self.current_player]['tokens'][color] -= requested_tokens[color]
            self.tokens[color] += requested_tokens[color]

        self.return_tokens = False

    def remove_card(self, card):
        card_idx = card.name
        # print(card_idx)
        if int(card['tier']) == 1:
            self.tier1 = self.tier1.drop(index=int(card_idx), errors='ignore')
        elif int(card['tier']) == 2:
            self.tier2 = self.tier2.drop(index=int(card_idx), errors='ignore')
        elif int(card['tier']) == 3:
            self.tier3 = self.tier3.drop(index=int(card_idx), errors='ignore')
        else:
            assert False, 'invalid tier'
   
    def show_cards(self):
		# Returns string versions of all visible cards on board
        shown_tier1 = self.tier1[-min(4, len(self.tier1)):].reset_index(drop=True)
        shown_tier2 = self.tier2[-min(4, len(self.tier2)):].reset_index(drop=True)
        shown_tier3 = self.tier3[-min(4, len(self.tier3)):].reset_index(drop=True)
        # print(shown_tier1)
        # print(shown_tier2)  
        # print(shown_tier3)
        # Convert cards to string representation
        str_tier1 = [self.card_to_colors(shown_tier1.iloc[i]) for i in range(len(shown_tier1))]
        str_tier2 = [self.card_to_colors(shown_tier2.iloc[i]) for i in range(len(shown_tier2))]
        str_tier3 = [self.card_to_colors(shown_tier3.iloc[i]) for i in range(len(shown_tier3))]

        return str_tier1 + str_tier2 + str_tier3
    # Chuyển đổi chỉ số của agent thành tên agent
    def _int_to_name(self, ind):
        return self.possible_agents[ind]
    
    def get_action_mask(self):
        # Ví dụ: tất cả các chỉ số từ 0 đến output_nodes - 1 đều hợp lệ
        valid_action = [0] * self.output_nodes
        for action in self.legal_actions:
            valid_action[action] = 1
            
        return np.array(valid_action)          

    

    def buy(self, card):
        # print("buy card", card)
		# If this card was reserved
        if self.card_to_colors(card) in self.show_reservations():
            print("buy reserved card")
            self.players[self.current_player]['tokens']['gold'] += 1
            reservations = self.players[self.current_player]['reservations']
            idx = card.index.tolist()[0]
            idxes = [i.index.tolist()[0] for i in reservations]
            to_pop = idxes.index(idx)
            self.players[self.current_player]['reservations'].pop(to_pop)
        
        else:
            print("buy card from board")
            # If player is buying card from board remove it
            self.remove_card(card)

        for color in self.colors:
            # Amount of this player's cards of certain color
            this_color_card = self.players[self.current_player]['cards'][color]
            print(color)
            # print("this color card of player", this_color_card)
            # If player doesnt have more cards of this color than needed
            if int(card[color]) > this_color_card:

                # Subtract missing tokens of this color from player and put it back on the board
                necessary_tokens = int(card[color]) - this_color_card
                # print("necessary tokens", necessary_tokens)
                if(self.players[self.current_player]['tokens'][color] >= necessary_tokens):
                    self.players[self.current_player]['tokens'][color] -= necessary_tokens
                    self.tokens[color] += necessary_tokens
                else: 
                    # If player has not enough tokens of this color
                    # he can use gold tokens to buy this card
                    necessary_tokens = necessary_tokens - self.players[self.current_player]['tokens'][color]
                    
                    self.players[self.current_player]['tokens']['gold'] -= necessary_tokens
                    self.tokens['gold'] += necessary_tokens
                    
                    self.tokens[color] += self.players[self.current_player]['tokens'][color]
                    self.players[self.current_player]['tokens'][color] = 0

        # Add card power (color) to player arsenal and card value to player score
        card_color = self.colors[int(card['color'])-1]
        self.players[self.current_player]['cards'][card_color] += 1
        self.players[self.current_player]['score'] += int(card['value'])
  
    def check_nobles(self):
		# Check if player that moved as the last could gain any nobleman
        this_cards = self.players[self.current_player]['cards']
        for nobleman in range(len(self.nobles)):
            this_nobleman = self.nobles[nobleman: nobleman+1]
            for color in self.colors:
                if this_cards[color] < int(this_nobleman.iloc[0][color]):  # updated line
                    break

            # Player can receive at most 1 nobel at each round which gives him 3 points
            else:
                self.players[self.current_player]['nobles'] += 1
                self.players[self.current_player]['score'] += NOBLEMAN_VALUE
                #self.nobles.drop(self.nobles[nobleman].index.tolist())
                self.remove_nobleman(this_nobleman)
                break

    def reserve(self, card):
        # Remove card and one gold token from board 
        # and add card to player reservations
        self.remove_card(card)  
        self.players[self.current_player]['reservations'].append(card)
        if self.tokens['gold'] > 0:
            self.players[self.current_player]['tokens']['gold'] += 1
            self.tokens['gold'] -= 1
    def human_to_action(self, interactive=True):
        while True:
            try:
                # Print the current available legal moves for the current player.
                print(f"Current available actions for player {self.agent_selection} are: {self.legal_actions}")

                if interactive:
                    print(self.tier1[0:4])
                    print(self.tier2[0:4])
                    print(self.tier3[0:4])
                    print(self.tokens)
                    print(self.players[self.current_player]['tokens'])
                    # Prompt the user to input the next move in either UCI string format or as an index.
                    choice = input(f"Enter the next move for player {self.agent_selection} (UCI format or index): ").strip()

                    # If the input is a digit, assume it is the action index.
                    if choice.isdigit():
                        action = int(choice)
                        # Check if the action is a legal move.
                        if action in self.legal_actions:
                            return action
                    else:
                        # If the input is not a digit, assume it is a UCI string.
                        # Convert the UCI string to a chess.Move object.
                        print(f"Nhập sai input, vui lòng nhập lại")
                else:
                    # If not in interactive mode, automatically select the first available legal action.
                    return self.legal_actions[0]
            except KeyboardInterrupt:
                # Handle user interruption (e.g., Ctrl+C).
                sys.exit(0)
            except Exception as e:
                # Handle any other exceptions, prompt the user to try again.
                print(f"Invalid input, please try again: {e}")
                
    def random_action(self):
        # print(self.legal_actions, self.agent_selection)
        return np.random.choice(self.legal_actions)
    
    @property
    def observation_space(self, agent) -> spaces.Space:
        return self._observation_space[agent]

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space
    
    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))
    
    @property
    def current_player(self):
        return self.current_player_index
    
    def close(self) -> None:
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        
    def __repr__(self) -> str:
        return "Splendor Env"
