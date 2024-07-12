"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
import argparse
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', default=100000, type=int, help='Number of steps')
    parser.add_argument('--n_eval_episodes', default=1e3, type=int, help='Number of steps')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--seed', default=1000, type=int, help='random seed')
    
    return parser.parse_args()

args = parse_args()

def main():
    train_env = gym.make('CustomHopper-source-v0')
    eval_env = gym.make('CustomHopper-source-v0') 
    if args.seed is not None:
        train_env.seed(args.seed)
    check_env(train_env)


    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    if args.seed is not None:
        model=PPO('MlpPolicy', train_env, seed=args.seed)
    else:
        model=PPO('MlpPolicy', train_env)
        
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    model.save("custom_hopper")
    del model

    model=PPO.load("custom_hopper", env=train_env)
    
    # if train and eval will be do on the same env --> eval_env=model.get_env(), otherwise insert another eval env
    mean_reward, std_reward=evaluate_policy(model, model.get_env(), n_eval_episodes=args.n_eval_episodes)

    env=model.get_env()
    obs=env.reset()
    for i in range(args.total_timesteps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

  

if __name__ == '__main__':
    main()
