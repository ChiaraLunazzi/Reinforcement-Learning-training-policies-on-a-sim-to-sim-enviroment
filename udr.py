
import gym
import argparse
import wandb
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from collections import ChainMap
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EveryNTimesteps
from algorithms.custom_callback import myCallback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_episodes', default=1000, type=int, help='Number of training episodes')
    parser.add_argument('--test_episodes', default=50, type=int, help='Number of test episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--train', default='target', type=str, help='Training enviroment')
    parser.add_argument('--test', default='target', type=str, help='Test enviroment')
    parser.add_argument('--env', default=[None, None], type=str, help='Enviroments') #[train_env, test_env], 
    parser.add_argument('--seed', default=1000, type=int, help='random seed')
    parser.add_argument('--lower_bound', default=[0.001, 3072, 32, 50], type=float, help='tuned parameters')
    parser.add_argument('--upper_bound', default=[0.001, 3072, 32, 20], type=float, help='tuned parameters')
    parser.add_argument('--width_factor', type=float, nargs='+', default=[0.1, 0.2, 0.3], help='Width factors for domain randomization')

    return parser.parse_args()

args = parse_args()


def create_env(train_env_name, test_env_name):
    train_env = gym.make(train_env_name)
    test_env = gym.make(test_env_name)
    
    train_env = Monitor(train_env)
    test_env = Monitor(test_env)

    return train_env, test_env

def domain_randomization(train_env, width_factor):
    print("Applying domain randomization...")
    train_env.set_random_parameters(width_factor).seed(arg.seed)
    return train_env   

def train_test(args, train_env, test_env, width_factor):
    # Setup callbacks
    callback=myCallback(train_env, width_factor)
    
    # Initialize PPO model
    if args.seed is not None:
        model = PPO("MlpPolicy", env=train_env, learning_rate=args.upper_bound[0], n_steps=args.upper_bound[1], batch_size=args.upper_bound[2], n_epochs=args.upper_bound[3], seed=args.seed)
    else:
        model=PPO("MlpPolicy", env=train_env, learning_rate=args.lower_bound[0], n_steps=args.lower_bound[1], batch_size=args.lower_bound[2], n_epochs=args.lower_bound[3])


    # Train the model
    model.learn(total_timesteps=2e3, reset_num_timesteps=False, callback=callback, progress_bar=True)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=args.test_episodes)

    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Log results to WandB
    wandb.log({
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "width_factor": width_factor
    })
    return train_env, test_env

def main():
    args = parse_args()

    train_env_name = f'CustomHopper-target-v0'
    test_env_name = f'CustomHopper-target-v0'

    train_env, test_env = create_env(train_env_name, test_env_name)
    if args.seed is not None:
        train_env.seed(args.seed)
        test_env.seed(args.seed)
    
    # Initialize WandB
    wandb.init(project="custom_hopper_project")

    for w in args.width_factor:
        train_test(args, train_env, test_env, w)
    
    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    main()
