#use callback to have better results

import gym
import argparse
import wandb
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from  stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.env_util import make_vec_env
from collections import ChainMap
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--test_episodes', default=50, type=int, help='Number of test episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--env', default=[None, None], type=str, help='Enviroments') #[train_env, test_env], 
    parser.add_argument('--seed', default=1000, type=int, help='random seed')
    parser.add_argument('--dictionary', default=dict(mean=None, learning_rate=None, batch_size=None, n_steps=None, n_epochs=None), type=dict, help='Tuned parameters')
    
    return parser.parse_args()

args = parse_args()


def create_env(train_env_name, test_env_name):
    if train_env_name is None or test_env_name is None:
        raise Exception('Enviroment should be spieciefied')

    args.env[0]=gym.make(train_env_name)
    args.env[1]=gym.make(test_env_name)
        
    args.env[0]=Monitor(args.env[0])
    args.env[1]=Monitor(args.env[1])
    
    # check if personal envs are correct
    check_env(args.env[0])
    check_env(args.env[1])

    

def train_test():
    wandb.init()
    config=wandb.config

    checkpoint_callback=CheckpointCallback(save_freq=500, save_path="./logs/")
    eval_callback=EvalCallback(args.env[1], best_model_save_path="./logs/", log_path="./logs/", eval_freq=500, deterministic=True, render=False)

    callback=CallbackList([checkpoint_callback, eval_callback])
    
    if args.seed is not None:
         model=PPO(policy='MlpPolicy', env=args.env[0], learning_rate=config.learning_rate, n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs, seed=args.seed)
    else:
         model=PPO(policy='MlpPolicy', env=args.env[0], learning_rate=config.learning_rate, n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs)
         
    model.learn(total_timesteps=2e3, callback=callback, progress_bar=True)
    
    mean_reward, _ = evaluate_policy(model=model, env=args.env[1], n_eval_episodes=args.test_episodes, deterministic=True, render=False) #set render=True to open glfw windows
    
    wandb.log({'mean_reward': mean_reward, 'learning_rate': config.learning_rate, 'n_steps': config.n_steps, 'batch_size': config.batch_size, 'n_epochs': config.n_epochs})
    
    if args.dictionary["mean"] is None or args.dictionary["mean"]< mean_reward:
        args.dictionary.update({'mean': mean_reward, 'learning_rate': config.learning_rate, 'batch_size': config.batch_size, 'n_steps': config.n_steps, 'n_epochs': config.n_epochs})
    

def main():
    train_env_name = f'CustomHopper-target-v0'
    test_env_name = f'CustomHopper-target-v0'
    create_env(train_env_name, test_env_name)
    if args.seed is not None:
        args.env[0].seed(args.seed)
        args.env[1].seed(args.seed)

    print('State space:', args.env[0].observation_space)
    print('Action space:', args.env[0].action_space)
    print('Dynamics parameters:', args.env[0].get_parameters())
    
    
    # Tuning hyperparameters with wandb sweep configuration
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'mean_reward',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [0.1, 0.01, 0.001, 0.0001]
	    },		
            'n_steps': {
                'values': [1024, 2048, 3072]
            },
            'batch_size': {
                'distribution': 'q_uniform',
                'q': 16,
                'min': 16,
                'max': 64
            },
            'n_epochs': {
                'values': [10, 20, 50]
            }
        }
    } 
    
    sweep_id = wandb.sweep(sweep_config, project='target-target-callback')   
    wandb.agent(sweep_id=sweep_id, function=train_test)
    
    print('mean_value best parameter:')
    print(args.dictionary)

    

if __name__ == '__main__':
    main()
