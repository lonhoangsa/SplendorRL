import pytest
from splendor_lightzero_env import SplendorLightZeroEnv

class TestSplendorEnvAuto:

    def test_random_play(self):
        env = SplendorLightZeroEnv()
        # obs = env.reset()
        print('Initial board state:')
        # env.render()

        done = False
        step = 0
        while not done:
            step += 1
            print(f'Step {step}:')
            print(env.legal_actions)
            # Player 1 (random action)
            action = env.random_action()
            print(f'Player 1 action: {action}')
            obs, reward, done, info = env._step(action)
            # self.check_step_outputs(obs, reward, done, info)
            # env.render()
            if done:
                self.print_game_result(reward, info, 'Player 1')
                break
            
            # Player 2 (random action)
            print(env.legal_actions)
            action = env.random_action()
            print(f'Player 2 action: {action}')
            obs, reward, done, info = env._step(action)
            # self.check_step_outputs(obs, reward, done, info)
            # env.render()
            if done:
                self.print_game_result(reward, info, 'Player 2')
                break
            
            # Player 3 (random action)
            print(env.legal_actions)
            action = env.random_action()
            print(f'Player 3 action: {action}')
            obs, reward, done, info = env._step(action)
            # self.check_step_outputs(obs, reward, done, info)
            # env.render()
            if done:
                self.print_game_result(reward, info, 'Player 3')
                break
            
            # Player 4 (random action)
            print(env.legal_actions)
            action = env.random_action()
            print(f'Player 4 action: {action}')
            obs, reward, done, info = env._step(action)
            # self.check_step_outputs(obs, reward, done, info)
            # env.render()
            if done:
                self.print_game_result(reward, info, 'Player 4')
                break
            
        env.close()
        
    def print_game_result(self, reward, info,  player_name):
        print("===================================")
        print(f"Game result for {player_name}:")
        print(f"Score: {reward}")
        print(f"Info: {info}")
        # print(f"Final state: {self.render()}")
        print("===================================")
        
    def test_random_vs_human(self):
        env = SplendorLightZeroEnv()
        # obs = env.reset()
        print('Initial board state:')
        # env.render()
        # env.print_pick_token_colors()
        done = False
        step = 0
        while not done:
            step += 1
            print(f'Step {step}:')
            
            # Player 1 (random action)
            action = env.human_to_action(interactive=True)
            print(f'Player 1 action: {action}')
            obs, reward, done, info = env.step(action)
            print(f'Player 1 reward: {reward}')
            print(obs)
            # self.check_step_outputs(obs, reward, done, info)
            # env.render()
            if done:
                self.print_game_result(reward, 'Player 1')
                break
            
            # Player 2 (random action)
            action = env.random_action()
            print(f'Player 2 action: {action}')
            obs, reward, done, info = env.step(action)
            print(f'Player 2 reward: {reward}')
            # self.check_step_outputs(obs, reward, done, info)
            # env.render()
            if done:
                self.print_game_result(reward, 'Player 2')
                break
            
            # Player 3 (random action)
            action = env.random_action()
            print(f'Player 3 action: {action}')
            obs, reward, done, info = env.step(action)
            print(f'Player 3 reward: {reward}')
            # self.check_step_outputs(obs, reward, done, info)
            # env.render()
            if done:
                self.print_game_result(reward, 'Player 3')
                break
            
            # Player 4 (random action)
            action = env.random_action()
            print(f'Player 4 action: {action}')
            obs, reward, done, info = env.step(action)
            print(f'Player 4 reward: {reward}')
            # self.check_step_outputs(obs, reward, done, info)
            # env.render()
            if done:
                self.print_game_result(reward, 'Player 4')
                break
            
        env.close()
            
if __name__ == "__main__":
    test = TestSplendorEnvAuto()
    test.test_random_play()
        