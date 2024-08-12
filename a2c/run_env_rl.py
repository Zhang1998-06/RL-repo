import numpy as np
import math
import torch
import random
from toy_env import Toy_env
from det_mdp import Model_MDP
from a2c_toy import PolNet, ValNet, AgentA2C

def choose_action(left_prob):
    samp_rand = random.random()
    # print('samp_rand', samp_rand)
    if samp_rand <= left_prob:
        action = -1
    else:
        action = 1

    return action

def run_toy(method_name, agent, gamma):
    Exp = Toy_env()
    cur_state = Exp.reset()         # [0, 1, 2, 3, 4]
    done = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_list = [cur_state]
    action_list = []
    reward_list = []
    mu_list = []
    var_list = []
    value_list = []

    if method_name == 'a2c_only':
        state_pos_max = 4

        while not done:
            states = torch.Tensor(cur_state).to(device)
            norm_states = states/torch.Tensor([state_pos_max, 1, 1])
            left_prob, left_prob_mu, left_prob_var = agent.get_action(norm_states)
            val = agent.get_value(norm_states)

            action = choose_action(left_prob)
            cur_state, rewards, done = Exp.step(cur_state, action)
            action_list.append(action)
            state_list.append(cur_state)
            reward_list.append(rewards)
            mu_list.append(left_prob_mu.data.cpu().numpy())
            var_list.append(left_prob_var.data.cpu().numpy())
            value_list.append(val)
        
        # print('state trans: ', np.array(state_list))
        print('final reward: ', sum(reward_list))
        print('steps', len(reward_list))
        reward_batch = np.array(reward_list)
        nex_state_v = np.array(state_list[1:])
        nex_val_batch = agent.get_value(nex_state_v)
        ref_vals_batch = gamma*reward_batch + np.ravel(nex_val_batch.data.cpu().numpy())

        return np.array(action_list), np.array(state_list[:-1]), \
               np.array(mu_list), np.array(var_list), ref_vals_batch, sum(reward_list)
