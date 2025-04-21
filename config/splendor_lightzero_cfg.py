from easydict import EasyDict
# ==============================================================================
# Main configuration for Splendor MuZero
# ==============================================================================
splendor_muzero_config = dict(
    exp_name='splendor_muzero_seed0',
    env=dict(
        env_id='splendor_lightzero',           # Tên môi trường đã đăng ký
        collector_env_num=8,                    # Số môi trường song song để thu thập dữ liệu
        evaluator_env_num=4,                    # Số môi trường song song để đánh giá
        n_evaluator_episode=10,                 # Số episode đánh giá trong evaluator
        collect_max_episode_steps=1000,         # Số bước tối đa khi thu thập
        eval_max_episode_steps=1000,            # Số bước tối đa khi đánh giá
        frame_stack_num=1,                      # Không stack frame (Splendor không cần)
        gray_scale=False,                       # Không cần grayscale (Splendor không dùng ảnh)
        scale=True,                             # Chuẩn hóa observation thành float32
        clip_rewards=False,                     # Không cắt reward
        episode_life=False,                     # Không có khái niệm "lives" trong Splendor
        env_type='board',                       # Loại môi trường: board game
        frame_skip=1,                           # Không skip frame
        stop_value=15,                          # Điểm dừng (WINNING_SCORE = 15)
        replay_path='./splendor_replays',       # Đường dẫn lưu replay
        save_replay=True,                       # Bật lưu replay
        channel_last=False,                     # Channel không ở cuối (Splendor không dùng ảnh)
        warp_frame=False,                       # Không cần crop frame
        manager=dict(
            shared_memory=False,                # Không cần shared memory
            reset_timeout=600,                  # Timeout khi reset
            max_retry=5,                        # Số lần thử lại tối đa
        ),
    ),
    policy=dict(
        model=dict(
            observation_shape=175,              # Tổng kích thước phẳng: 36 (current_player) + 96 (tiers) + 6 (tokens) + 25 (nobles) + 12 (opponents)
            action_space_size=73,               # Số hành động (output_nodes từ code)
            continuous_action_space=False,      # Không gian hành động rời rạc
            num_res_blocks=4,                   # Số residual blocks trong mạng
            downsample=False,                   # Không downsample (Splendor không dùng ảnh)
            norm_type='BN',                     # Batch Normalization
            num_channels=128,                   # Số kênh trong mạng
            support_scale=15,                   # Giá trị hỗ trợ tối đa (dựa trên WINNING_SCORE)
            bias=True,                          # Sử dụng bias trong layers
            discrete_action_encoding_type='one_hot',  # Mã hóa hành động dạng one-hot
            self_supervised_learning_loss=False,      # Không dùng SSL cho Splendor
            image_channel=0,                    # Không dùng kênh ảnh
            frame_stack_num=1,                  # Không stack frame
            gray_scale=False,                   # Không dùng grayscale
            use_sim_norm=False,                 # Không dùng SimNorm
            use_sim_norm_kl_loss=False,         # Không dùng KL loss
            res_connection_in_dynamics=True,    # Residual connection trong dynamics
        ),
        cuda=True,                              # Sử dụng GPU nếu có
        multi_gpu=False,                        # Không dùng multi-GPU mặc định
        use_wandb=False,                        # Không dùng Weights & Biases mặc định
        mcts_ctree=False,                       # Không dùng C++ MCTS (dùng Python mặc định)
        env_type='board',                       # Loại môi trường: board game
        action_type='fixed',                    # Loại hành động: cố định (Splendor có action space cố định)
        game_segment_length=50,                 # Độ dài đoạn game cho MCTS
        cal_dormant_ratio=False,                # Không tính tỷ lệ neuron "ngủ"
        use_augmentation=False,                 # Không dùng data augmentation
        augmentation=[],                        # Danh sách augmentation trống
        random_collect_episode_num=100,         # Số episode thu thập ngẫu nhiên ban đầu
        update_per_collect=200,                 # Số lần cập nhật sau mỗi lần thu thập
        batch_size=256,                         # Kích thước batch
        optim_type='Adam',                      # Loại optimizer
        reanalyze_ratio=0.5,                    # Tỷ lệ reanalyze
        reanalyze_noise=True,                   # Thêm noise khi reanalyze
        reanalyze_batch_size=64,                # Kích thước batch khi reanalyze
        reanalyze_partition=1.0,                # Reanalyze toàn bộ buffer
        learning_rate=0.001,                    # Tốc độ học ban đầu
        num_simulations=50,                     # Số mô phỏng MCTS
        reward_loss_weight=1.0,                 # Trọng số loss cho reward
        policy_loss_weight=1.0,                 # Trọng số loss cho policy
        value_loss_weight=1.0,                  # Trọng số loss cho value
        ssl_loss_weight=0.0,                    # Không dùng SSL loss
        n_episode=16,                           # Số episode song song trong collector
        eval_freq=100,                          # Tần suất đánh giá
        replay_buffer_size=10000,               # Kích thước replay buffer
        target_update_freq=200,                 # Tần suất cập nhật target network
        grad_clip_value=5.0,                    # Giá trị cắt gradient
        discount_factor=0.99,                   # Hệ số chiết khấu
        td_steps=5,                             # Số bước TD
        num_unroll_steps=5,                     # Số bước unroll trong MuZero
        # ==============================================================================
        # Bắt đầu các tham số thường xuyên thay đổi
        # ==============================================================================
        collector_env_num=8,                    # Số môi trường thu thập
        evaluator_env_num=4,                    # Số môi trường đánh giá
        max_env_step=1000000,                   # Số bước môi trường tối đa
        # ==============================================================================
        # Kết thúc các tham số thường xuyên thay đổi
        # ==============================================================================
    ),
    main_config=dict(
        seed=0,                                 # Seed để tái lập kết quả
        env_iter=1000000,                       # Tổng số bước môi trường
        policy_iter=10000,                      # Tổng số lần cập nhật policy
        save_freq=500,                          # Tần suất lưu checkpoint
        eval_freq=100,                          # Tần suất đánh giá
        log_freq=50,                            # Tần suất ghi log
    ),
)

splendor_muzero_config = EasyDict(splendor_muzero_config)
main_config = splendor_muzero_config

# ==============================================================================
# Create configuration for environment and policy
# ==============================================================================
create_config = dict(
    env=dict(
        type='splendor_lightzero',          # Tên class môi trường
        import_names=['zoo.board_games.splendor.env.splendor_lightzero_env'],          # Nếu môi trường định nghĩa trong file này
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',                      # Loại policy: MuZero
        import_names=['lzero.policy.muzero'],  # Đường dẫn import policy MuZero
    ),
)
create_config = EasyDict(create_config)

# ==============================================================================
# Main entry point
# ==============================================================================
if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=main_config.main_config.seed, max_env_step=main_config.policy.max_env_step)