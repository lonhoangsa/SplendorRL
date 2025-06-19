from AlphaZero.trainer import AlphaZeroTrainer
from AlphaZero.config import complex_config
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train AlphaZero agent with complex architecture')
    parser.add_argument('--iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load model from')
    parser.add_argument('--save_interval', type=int, default=5, help='Number of iterations between model saves')
    parser.add_argument('--eval_interval', type=int, default=10, help='Number of iterations between evaluations')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no_continue', action='store_true', help='Start training from scratch even if loading a model')
    args = parser.parse_args()

    # Create trainer with complex architecture config
    trainer = AlphaZeroTrainer(config=complex_config, debug=args.debug)
    
    # Run training
    trainer.run(
        iterations=args.iterations,
        load_path=args.load_path,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        continue_training=not args.no_continue
    )

if __name__ == '__main__':
    main() 