"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym
import wandb

from env.custom_hopper import *
from algorithms.agent_reinforce import Agent, Policy
#from algorithms.agent_critic import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--seed', default=1000, type=int, help='random seed')
    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	#env = gym.make('CustomHopper-target-v0')
	if args.seed is not None:
		env.seed(args.seed)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim, args.seed)
	agent = Agent(policy, device=args.device)
	wandb.init(project='actor-critic algorithm on source')

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		wandb.log({'train_reward': train_reward})
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode+1)
			print('Episode return:', train_reward)
		agent.update_policy()	

	torch.save(agent.policy.state_dict(), "model.mdl")

	

if __name__ == '__main__':
	main()
