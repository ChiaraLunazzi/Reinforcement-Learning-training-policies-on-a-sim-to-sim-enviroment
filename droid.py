#REMEMBER: TARGET DOMAIN--> REAL WORLD!!!!!

import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
from env.custom_hopper import *
from env.mujoco_env import MujocoEnv
from stable_baselines3 import PPO
from  stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', default=1000, type=int, help='Number of learning steps')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--seed', default=1000, type=int, help='random seed')
    return parser.parse_args()

args = parse_args()

#normalize the trajectories
def normalize(trajectories):
    min_value = np.min(trajectories, axis=0)
    max_value = np.max(trajectories, axis=0)
    
    norm_trajectories = (trajectories - min_value) / (max_value - min_value)
    return norm_trajectories

def train(env, parameters, n_timesteps):
    learning_rate = parameters[0]
    n_steps = parameters[1]
    batch_size = parameters[2]
    n_epochs = parameters[3]
    
    model=PPO(policy='MlpPolicy', env=env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, seed=args.seed)
    model.learn(total_timesteps= n_timesteps, progress_bar=False)
    
    env=model.get_env()
    obs=env.reset()
    pos=[]
 
    for i in range(n_timesteps):
        action, states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        xpos=env.envs[0].get_body_com("foot")
        pos.append([xpos[0], xpos[2]])
    return pos, model

def evaluate(eval_env, model, n_evaluation_steps):
    mean_reward, std_reward=evaluate_policy(model, eval_env, n_eval_episodes=n_evaluation_steps, deterministic=True, render=True)
    return mean_reward

def get_estimator(env):
    mu=np.mean(env.get_parameters())
    sigma=np.std(env.get_parameters())
    return mu, sigma  

def randomize(sim_env, mu, sigma):
    print("Applying domain randomization...")
    sim_env.set_random_parameters(0.5, mu, sigma, True)
    sim_env.seed(args.seed)
      

def plot(t1, t2):
    plt.plot(t1[:,0], t1[:,1], color='r', label = 'target')
    plt.plot(t2[:,0], t2[:,1], color='g', label = 'sim')   
    plt.legend() 
    plt.show()

def main():
    # real world simulation
    target_env = gym.make('CustomHopper-target-v0')
    target_env = Monitor(target_env)
    sim_env = gym.make('CustomHopper-source-v0')
    sim_env = Monitor(sim_env)
    
    if args.seed is not None:
        target_env.seed(args.seed)
        sim_env.seed(args.seed)
        
    # collect the trajectories over target_env
    target_parameters=[0.001, 3072, 32, 20]
    
    # number of time we will randomize masses
    M = 100
    
    # penality and discount factor
    b = 100
    c = 0.5
    
    #tollerance
    tol = 1e-2
    
    mu, sigma = get_estimator(target_env)
    
    target_trajectories, _ = train(target_env, target_parameters, n_timesteps = 2000)
    target_trajectories = np.array(target_trajectories)
    #target_trajectoris = normalize (target_trajectories)
    N = len(target_trajectories)
    sim_trajectories, _ = train(sim_env, target_parameters, n_timesteps = 2000)
    sim_trajectories = np.array( sim_trajectories)
    #sim_trajectories = normalize(sim_trajectories)
    distance = np.linalg.norm(target_trajectories - sim_trajectories)
    
    print(sim_env.get_parameters())
    
    print(f'original distance : {distance}')
    
    while True:
        Jprev = 'Null'
        for _ in range(M):
            randomize(sim_env, mu, sigma)
            trajectories, model = train(sim_env, target_parameters, n_timesteps = 2000)
            trajectories = np.array(trajectories)
            #trajectories = normalize(trajectories)
            J = 1/N*sum(trajectories - np.power(target_trajectories, n+1) + c*b for n in range(N))
            if Jprev == 'Null' or J < Jprev:
                sim_trajectories = trajectories
                best_parameters = sim_env.get_parameters()
        sim_env.override_masses(best_parameters)
        mu, sigma = get_estimator(sim_env)
        print(sim_env.get_parameters())
        distance = np.linalg.norm(target_trajectories - sim_trajectories)
        print(f'actual distance : {distance}')
        if distance < tol:
            mean_reward = evaluate(target_env, model, n_eval_steps = 50)
            print(mean_reward)
            plot(target_trajectories, sim_trajectories)
            break
            
            
    
    
    
if __name__ == '__main__':
    main()
    
