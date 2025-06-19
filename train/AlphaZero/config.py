# config for AlphaZero

default_config = {
    'num_simulations': 20,
    'cpuct': 1.0,
    'batch_size': 2048,
    'lr': 0.005,
    'weight_decay': 0.0001,
    'epochs': 15000,
    'replay_size': 100000,
    'games_per_iteration': 30,
    'eval_games': 5,
    'debug_mode': False,
    'min_score_threshold': 15,
    'efficiency_bonus': 0.4,
    
    'progressive_widening': True,
    'temperature': 1,
    'temperature_decay': 0.99,
    'min_temperature': 0.01,
    'curriculum_learning': False,
    'curriculum_stages': [
        {'min_score': 5},   # Giai đoạn 1: Tập trung vào việc mua thẻ cơ bản
        {'min_score': 10},  # Giai đoạn 2: Bắt đầu tập trung vào Noble tiles
        {'min_score': 15}   # Giai đoạn 3: Chiến lược nâng cao
    ],
    'pin_memory': True,
    'num_workers': 4,
    'prefetch_factor': 2,
    'non_blocking': True,
    'max_depth': 500
}   