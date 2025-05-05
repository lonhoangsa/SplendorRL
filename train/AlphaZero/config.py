# config for AlphaZero

default_config = {
    'num_simulations': 40,
    'cpuct': 1.5,
    'batch_size': 2048,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'epochs': 100,
    'replay_size': 10000,
    'games_per_iteration': 30,
    'eval_games': 20,
    'debug_mode': False,
    'min_score_threshold': 10,
    'noble_bonus': 3,
    'set_bonus': 2,
    'efficiency_bonus': 1,
    'progressive_widening': True,
    'temperature': 1.0,
    'temperature_decay': 0.99,
    'min_temperature': 0.1,
    'curriculum_learning': False,
    'curriculum_stages': [
        {'min_score': 5},   # Giai đoạn 1: Tập trung vào việc mua thẻ cơ bản
        {'min_score': 10},  # Giai đoạn 2: Bắt đầu tập trung vào Noble tiles
        {'min_score': 15}   # Giai đoạn 3: Chiến lược nâng cao
    ]
}   