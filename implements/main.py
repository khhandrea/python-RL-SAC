from trainer import Trainer

import gymnasium as gym
from matplotlib import pyplot as plt
from torch import tensor, float32, load

from argparse import ArgumentParser
from typing import Tuple

if __name__ == '__main__':
    parser = ArgumentParser(description='Soft Actor-Critic args')
    parser.add_argument('--env_name', default='Ant-v4')
    parser.add_argument('--healthy_min', type=float, default=0.3)
    parser.add_argument('--pool_size', type=int, default=1_000_000)
    parser.add_argument('--hidden_layer_nodes', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--scale_reward', type=int, default=5)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--start_step', type=int, default=10000)
    parser.add_argument('--num_step', type=int, default=500000)
    parser.add_argument('--evaluate_episode', type=int, default=10)
    parser.add_argument('--evaluate_term', type=int, default=30)
    parser.add_argument('--load', default='')
    parser.add_argument('--load_and_train', default='')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--skip_demo', action='store_true')
    args = parser.parse_args()

    if args.load and args.load_and_train:
        raise Exception("expected only one argument between '--load' and 'load_and_train'")

    trainer = Trainer(args)    

    # Load models
    if args.load:
        trainer.load_agent(args.load)
    elif args.load_and_train:
        trainer.load_agent(args.load_and_train)

    # Train
    if (not args.load) or args.load_and_train:
        trainer.train()
        
    # Demonstrates
    if not args.skip_demo:
        trainer.test()