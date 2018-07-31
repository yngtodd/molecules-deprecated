# Code adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


#parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
#parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                    help='discount factor (default: 0.99)')
#parser.add_argument('--seed', type=int, default=543, metavar='N',
 #                   help='random seed (default: 543)')
#parser.add_argument('--render', action='store_true',
 #                   help='render the environment')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
  #                  help='interval between training status logs (default: 10)')
#args = parser.parse_args()

# TODO: Change to RMSD environment
# Make own env
env = gym.make('CartPole-v0')
env.seed(args.seed) # Take out or put in random pdb
# For reproducibility to initialize starting weights
torch.manual_seed(42) 


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
	input_dim = -1 # Place holder
        self.dense1 = nn.Linear(4, 128)
        self.direction = nn.Linear(128, 2)
	self.magnitude = nn.Linear(128, 4)
	
        self.saved_log_probs_direction = []
	self.saved_log_probs_magnitude = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.dense1(x))
        action_direction = self.direction(x)
	action_magnitude = self.magnitude(x)
	scores_direction = F.softmax(action_direction, dim=1)
	scores_magnitude = F.softmax(action_magntiude, dim=1)
        return scores_direction, scores_magnitude


# TODO: Double check params with Todd
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
# Randomly choose an eps to normalize rewards
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    # TODO: ask about Todd about state variable
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs_direction, probs_magnitude = policy.forward(state)
    m_direction = Categorical(probs_direction)
    m_magnitude = Categorical(probs_magnitude)
    action_direction = m_direction.sample()
    action_magnitude = m_magnitude.sample()
    policy.saved_log_probs_direction.append(m_direction.log_prob(action_direction))
    policy.saved_log_probs_magnitude.append(m_magnitude.log_prob(action_magnitude))
    return action_direction.item(), action_magnitude.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    # Take out running_reward
    running_reward = 10
    for i_episode in count(1):
	# Reset pole to starting conditions
	# Don't need
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            #if args.render:
            #    env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
