import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from env.custom_hopper2 import *
from env.mujoco_env import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
#from utils2 import estimate_parameter, set_hyperparameters, train, optimal_parameter_PPO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', default=1000, type=int, help='Number of learning steps')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--masses', default=[], type=float, help='original masses')
    parser.add_argument('--seed', default=1000, type=int, help='random seed')
    return parser.parse_args()

args = parse_args()

#normalize the trajectories
def normalize(trajectories):
    min_value = np.min(trajectories, axis=0)
    max_value = np.max(trajectories, axis=0)
    
    norm_trajectories = (trajectories - min_value) / (max_value - min_value)
    return norm_trajectories
    

def train(train_env, parameters, n_timesteps):
    learning_rate = parameters[0]
    n_steps = parameters[1]
    batch_size = parameters[2]
    n_epochs = parameters[3]
    
    model = PPO(policy='MlpPolicy', env=train_env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, seed=args.seed)
    model.learn(total_timesteps=n_timesteps, progress_bar=False)
    
    env = model.get_env()
    obs = env.reset()
    pos = []
 
    for i in range(n_timesteps):
        action, states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        xpos = env.envs[0].get_body_com("torso")
        pos.append([xpos[0], xpos[2]])
    return pos, model

def evaluate(eval_env, model, n_eval_episodes, render=False):
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True, render=render)
    return mean_reward

def get_estimator(env):
    mu = np.mean(env.get_parameters())
    sigma = np.std(env.get_parameters())
    return mu, sigma  

def randomize_mass(env, mu, sigma, n_masses):
    print("Applying mass randomization...")
    width_factor = 0.5
    bounds = [(mass * (1 - width_factor), mass * (1 + width_factor)) for mass in args.masses]
    masses = []
    for _ in range(n_masses):
        m =[truncnorm.rvs((low-mu)/sigma, (high-mu)/sigma, loc = mu, scale = sigma, size = 1) for low, high in bounds][1:]
        m.insert(0, args.masses[0])
        masses.append(m)
    return masses

def get_distance(t1, t2):
    if len(t1) != len(t2):
        raise Exception("trajectories must have the same length")   
    distance = np.zeros(len(t1))
    k = 0
    for p1, p2 in zip(t1,t2):
        distance[k] = np.linalg.norm(p1-p2)
        k +=1
    return np.max(distance)
    
def plot(t1, t2, title):
    plt.plot(t1[:, 0], t1[:, 1], color='r', label='target')
    plt.plot(t2[:, 0], t2[:, 1], color='g', label='sim')   
    plt.legend() 
    plt.title(title)
    plt.show()

def main():
    # real world simulation
    target_env = gym.make('CustomHopper-target-v0')
    target_env = Monitor(target_env)
    sim_env = gym.make('CustomHopper-source-v0')
    sim_env = Monitor(sim_env)
    
    if args.seed is not None:
        target_env.seed(args.seed)
        # sim_env.seed(args.seed)
        
    target_parameters = [0.001, 3072, 32, 20]
    sim_parameters = [0.001, 3072, 32, 50]
    
    # number of time we will randomize masses
    M = 100
    
    # penalty and discount factor
    b = 100
    c = 0.5
    
    # tolerance
    tol = 1e-2
    
    mu, sigma = get_estimator(target_env)
    
    # collect the trajectories over target_env
    target_trajectories, _ = train(target_env, target_parameters, n_timesteps=2000)
    target_trajectories = np.array(target_trajectories)
    target_trajectories = normalize(target_trajectories)
    N = len(target_trajectories)
    
    sim_trajectories, best_model = train(sim_env, sim_parameters, n_timesteps=2000)
    sim_trajectories = np.array(sim_trajectories)
    sim_trajectories = normalize(sim_trajectories)
    
    args.masses = sim_env.get_parameters()
    
    prev_mean_reward = evaluate(target_env, best_model, n_eval_episodes=50)
    mean_reward = 0
    
    distance = get_distance(sim_trajectories, target_trajectories)
    
    f = open("droid.txt",'w')
    
    print(f"Original masses of the sim environment: {args.masses} \t Original distance: {distance} \t Original mean reward: {prev_mean_reward}", file=f)
    print(f"Original masses of the sim environment: {args.masses}")
    print(f'Original distance: {distance}')
    print(f'Original mean reward: {prev_mean_reward}')
    
    attempts = 0
    n_max = 100  # we try to align the trajectories for n_max times 

    while attempts < n_max:
        Jprev = float('inf')
        if mean_reward >= prev_mean_reward:
            prev_mean_reward = mean_reward
            args.masses = sim_env.get_parameters()
            best_model = model
        masses = randomize_mass(sim_env, mu, sigma, M)      
        for mass in masses:
            sim_env.set_parameters(mass)
            trajectories, model = train(sim_env, sim_parameters, n_timesteps=2000)
            trajectories = np.array(trajectories)
            trajectories = normalize(trajectories)
            J = 1/N *sum((np.linalg.norm(trajectories - np.power(target_trajectories, n+1))) for n in range (N))
            if J < Jprev:
                sim_trajectories = trajectories
                best_parameters = sim_env.get_parameters()
                Jprev = J
        sim_env.set_parameters(best_parameters)
        mu, sigma = get_estimator(sim_env)
        print(f"New masses: {sim_env.get_parameters()}")
        distance = get_distance(sim_trajectories, target_trajectories)
        mean_reward = evaluate(target_env, model, n_eval_episodes=50)
        print(f'Actual distance: {distance}\t Actual mean reward: {mean_reward}', file = f)
        print(f'Actual distance: {distance}\t Actual mean reward: {mean_reward}')
        attempts += 1
        if distance < tol:
            best_model = model
            break
    mean_reward = evaluate(target_env, best_model, n_eval_episodes=50, render=True)
    print(f"Best mean reward: {mean_reward}")
    plot(target_trajectories, sim_trajectories, "convergence")
if __name__ == '__main__':
    main()

