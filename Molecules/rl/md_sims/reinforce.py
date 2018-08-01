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

    
class reinforce(object):
    def __init__(self):
        # For reproducibility to initialize starting weights
        torch.manual_seed(42)
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        # Randomly choose an eps to normalize rewards
        self.eps = np.finfo(np.float32).eps.item()
        self.env = environment()
        
        
        # Build initial directories
        if not os.path.exists("./results"):
            os.mkdir("./results", 0755)
        if not os.path.exists("./results/final_output"):
            os.mkdir("./results/final_output")
        if not os.path.exists("./results/final_output/intermediate_data"):
            os.mkdir("./results/final_output/intermediate_data")
            
        
    def select_action(self, state):
        # TODO: ask about Todd about state variable
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs_direction, probs_magnitude = self.policy.forward(state)
        m_direction = Categorical(probs_direction)
        m_magnitude = Categorical(probs_magnitude)
        action_direction = m_direction.sample()
        action_magnitude = m_magnitude.sample()
        self.policy.saved_log_probs_direction.append(m_direction.log_prob(action_direction))
        self.policy.saved_log_probs_magnitude.append(m_magnitude.log_prob(action_magnitude))
        return action_direction.item(), action_magnitude.item()
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.rewards[:]
        del policy.saved_log_probs[:]

    def main():
        path = "./results/iteration_rl_"
        if not os.path.exists(path + "%i" % 0):
            os.mkdir(path + "%i" % 0, 0755)
        path_1 = path + "%i/sim_%i_%i/" % (0,0,0)
        if not os.path.exists(path_1):
            os.mkdir(path_1, 0755)
            os.mkdir(path_1 + "/cluster", 0755)
            os.mkdir(path_1 + "/pdb_data", 0755)
    
        state = env.intial_state(path)
        for i_episode in count(1):
            if not os.path.exists(path + "%i" % i_episode):
                os.mkdir(path + "%i" % i_episode, 0755)
            path_1 = path + "%i/sim_%i_%i/" % (i_episode, i_episode, 0)
            if not os.path.exists(path_1):
                os.mkdir(path_1, 0755)
                os.mkdir(path_1 + "/cluster", 0755)
                os.mkdir(path_1 + "/pdb_data", 0755)
                
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            if done:
                break

        
            finish_episode()
        
        
        
        


if __name__ == '__main__':
    ren = reinforce()
    ren.main()
