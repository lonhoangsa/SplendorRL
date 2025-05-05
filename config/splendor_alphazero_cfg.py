from easydict import EasyDict

# ==============================================================================
# Main configuration for Splendor AlphaZero
# ==============================================================================
splendor_alphazero_config = dict(
    exp_name='splendor_alphazero_seed0',
    env=dict(
        env_id='splendor_lightzero',              # Environment ID
        collector_env_num=8,                      # Number of parallel environments for data collection
        evaluator_env_num=4,                      # Number of parallel environments for evaluation
        n_evaluator_episode=10,                   # Number of evaluation episodes
        collect_max_episode_steps=1000,          # Maximum steps for data collection episodes
        eval_max_episode_steps=1000,             # Maximum steps for evaluation episodes
        frame_stack_num=1,                        # No frame stacking needed for Splendor
        gray_scale=False,                         # No grayscale conversion needed
        scale=True,                               # Normalize observations
        clip_rewards=False,                       # Don't clip rewards
        episode_life=False,                       # No concept of "lives" in Splendor
        env_type='board',                         # Board game environment
        frame_skip=1,                             # No frame skipping
        stop_value=15,                            # Stop when score reaches 15 (WINNING_SCORE)
        replay_path='./splendor_replays',         # Path to save replays
        save_replay=True,                         # Save replays
        channel_last=False,                       # Channel not last (Splendor doesn't use images)
        warp_frame=False,                         # No need to warp frames
    ),
    
    # AlphaZero-specific parameters
    alphazero=dict(
        # Neural network parameters
        model=dict(
            # input_dim will be calculated at runtime from the environment's actual state
            action_size=73,                       # Number of possible actions (output_nodes from env)
            num_res_blocks=10,                    # Number of residual blocks in network
            num_channels=256,                     # Number of channels in residual blocks
        ),
        
        # MCTS parameters
        mcts=dict(
            num_simulations=800,                  # Number of MCTS simulations per move
            c_puct=5.0,                           # Exploration constant
            dirichlet_alpha=0.3,                  # Dirichlet noise alpha for exploration
            dirichlet_epsilon=0.25,               # Dirichlet noise weight (exploration vs prior policy)
            mcts_batch_size=8,                    # Batch size for MCTS simulations
        ),
        
        # Training parameters
        training=dict(
            batch_size=256,                       # Batch size for neural network training
            epochs=1,                             # Number of epochs per optimization step
            learning_rate=0.001,                  # Learning rate
            weight_decay=1e-4,                    # L2 regularization
            grad_clip=5.0,                        # Gradient clipping
            momentum=0.9,                         # SGD momentum
            replay_buffer_size=100000,            # Size of replay buffer
        ),
        
        # Self-play parameters
        self_play=dict(
            num_self_play_games=5000,             # Number of self-play games to generate
            num_training_iterations=100,          # Number of training iterations
            temperature_init=1.0,                 # Initial temperature for action selection
            temperature_final=0.1,                # Final temperature for action selection
            temperature_threshold=10,             # Move threshold for temperature adjustment
            checkpoint_frequency=100,             # Frequency of model checkpoints
            validation_frequency=20,              # Frequency of validation
            num_validation_games=40,              # Number of validation games
            virtual_loss=3.0,                     # Virtual loss for MCTS
            save_self_play_data=True,             # Save self-play data
            self_play_data_directory='./self_play_data', # Directory for self-play data
        ),
    ),
    
    # Training process parameters
    main_config=dict(
        seed=0,                                   # Random seed for reproducibility
        total_iterations=1000,                    # Total training iterations
        log_frequency=10,                         # Logging frequency
        save_frequency=100,                       # Saving frequency
        eval_frequency=50,                        # Evaluation frequency
        max_env_step=10000000,                    # Maximum environment steps
    ),
)

splendor_alphazero_config = EasyDict(splendor_alphazero_config)
main_config = splendor_alphazero_config 