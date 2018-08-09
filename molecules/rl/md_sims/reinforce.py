# Code adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

from environment import environment
import numpy as np
from itertools import count
import os

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
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class Policy(nn.Module):
    def __init__(self, input_dim):
        super(Policy, self).__init__() 
        self.dense1 = nn.Linear(input_dim, 32)
        self.direction = nn.Linear(32, 2)
        self.magnitude = nn.Linear(32, 4)

        self.saved_log_probs_direction = []
        self.saved_log_probs_magnitude = []
        self.rewards = []

    def forward(self, x):
        print('x input:', x)
        print('self.dense1(x):', self.dense1(x))
        print('F.relu(self.dense1(x)):', F.relu(self.dense1(x)))
        x = F.relu(self.dense1(x))
        print('x after relu:', x)
        action_direction = self.direction(x)
        action_magnitude = self.magnitude(x)
        print('action_direction:', action_direction)
        print('action_magnitude:', action_magnitude)
        scores_direction = F.softmax(action_direction, dim=1)
        scores_magnitude = F.softmax(action_magnitude, dim=1)
        return scores_direction, scores_magnitude

    
class reinforce(object):
    def __init__(self, sim_steps=20000, traj_out_freq=100, episodes=20, output_dir=None):
        # For reproducibility to initialize starting weights
        torch.manual_seed(459)
        self.sim_steps = sim_steps
        self.traj_out_freq = traj_out_freq
        self.episodes = episodes
        self.policy = Policy(input_dim=sim_steps/traj_out_freq)
        self.policy.apply(init_weights)
        self.optimizer = optim.SGD(self.policy.parameters(), lr=1e-8)
        # Randomly choose an eps to normalize rewards
        self.eps = np.finfo(np.float32).eps.item()
        if not os.path.exists(output_dir):
            raise Exception("Path " + str(output_dir) + " does not exist!")
        self.output_dir = output_dir
        
        # Build initial directories
        if not os.path.exists(self.output_dir + "/results"):
            os.mkdir(self.output_dir + "/results", 0755)
        if not os.path.exists(self.output_dir + "/results/final_output"):
            os.mkdir(self.output_dir + "/results/final_output")
        if not os.path.exists(self.output_dir + "/results/final_output/intermediate_data"):
            os.mkdir(self.output_dir + "/results/final_output/intermediate_data")

        self.env = environment(cvae_weights_path="../model_150.dms",
                               sim_steps=self.sim_steps,
                               traj_out_freq=self.traj_out_freq,
                               output_dir=self.output_dir)
        
    def select_action(self, state):
        # TODO: ask about Todd about state variable
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs_direction, probs_magnitude = self.policy.forward(state)
        print("prob dir:",probs_direction)
        print("prob mag:",probs_magnitude)
        m_direction = Categorical(probs_direction)
        m_magnitude = Categorical(probs_magnitude)
        action_direction = m_direction.sample()
        action_magnitude = m_magnitude.sample()
        self.policy.saved_log_probs_direction.append(m_direction.log_prob(action_direction))
        self.policy.saved_log_probs_magnitude.append(m_magnitude.log_prob(action_magnitude))

        # Selecting new RMSD threshold
        dirs = [-1, 1]
        direction = dirs[action_direction.item()]
        mags = [0.1, 0.2, 0.5, .9]
        magnitude = mags[action_magnitude.item()]

        return direction*magnitude
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        rewards = []
        gamma = 0.5
        for r in self.policy.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        #for r in self.policy.rewards:
        #    R = r + gamma * R
        #    rewards.append(R)
            
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        
        for log_prob_dir, log_prob_mag, reward in zip(self.policy.saved_log_probs_direction, self.policy.saved_log_probs_magnitude, rewards):
            policy_loss.append(-(log_prob_dir + log_prob_mag) * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs_direction[:]
        del self.policy.saved_log_probs_magnitude[:]

    def main(self):
        path = self.output_dir + "/results/iteration_rl_"
        if not os.path.exists(path + "%i" % 0):
            os.mkdir(path + "%i" % 0, 0755)
        path_1 = path + "%i/sim_%i_%i/" % (0,0,0)
        if not os.path.exists(path_1):
            os.mkdir(path_1, 0755)
            os.mkdir(path_1 + "/cluster", 0755)
            os.mkdir(path_1 + "/pdb_data", 0755)
    
        state = self.env.initial_state(path_1)
        for i_episode in count(1):
            # Create Directories
            if not os.path.exists(path + "%i" % i_episode):
                os.mkdir(path + "%i" % i_episode, 0755)
            # TODO: Set 3 as a user defined variable
            for j_cycle in range(1):
                path_1 = path + "%i/sim_%i_%i/" % (i_episode, i_episode, j_cycle)
                if not os.path.exists(path_1):
                    os.mkdir(path_1, 0755)
                    os.mkdir(path_1 + "/cluster", 0755)
                    os.mkdir(path_1 + "/pdb_data", 0755)
                
                print("state shape before select_action:", state.shape)
                action = self.select_action(state)
                print("state shape after select_action:", state.shape)
                state, reward, done = self.env.step(action, path_1, i_episode, j_cycle)
                print('\n\n\n\n')
                print('reward:',reward)
                print('\n\n\n\n')

                self.policy.rewards.append(reward)
                if done:
                    break
            if (j_cycle < 2) or i_episode == self.episodes:
                break

            for name, param in self.policy.named_parameters():
                if param.requires_grad:
                    print('Before finish name param.data:',name, param.data)
            self.finish_episode()
            print('After finish self.policy.parameters():', self.policy.parameters())
            for name, param in self.policy.named_parameters():
                if param.requires_grad:
                    print('After finish name param.data:',name, param.data)
        for i in range(1, i_episode + 1):
             print("print %i episode" %i) 
             # TODO: update 3 to user defined variable
             for j in range(1):       
                 self.env.plot_intermediate_episode(self.env.output_dir + "/results/final_output/intermediate_data/", 
                                                    self.env.output_dir + "/results/final_output/", i, j, 'Intermediate')
        


#if __name__ == '__main__':
#    ren = reinforce#()
#    ren.main()
