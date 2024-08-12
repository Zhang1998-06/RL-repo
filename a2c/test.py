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


def test_method(method_name):

    Exp = Toy_env()
    cur_state = Exp.reset()         # [0, 1, 2, 3, 4]
    done = 0

    state_list = [cur_state]
    action_list = []
    reward_list = []
    mu_list = []
    var_list = []
    value_list = []

    if method_name == 'model_base':
        init_pos = 1
        goal_pos = 0
        mid_pos = 4
        mdp = Model_MDP(init_pos, goal_pos, mid_pos)

        while not done:

            left_prob = mdp.choose_prob(cur_state)
            # print('left prob', left_prob)
            action = mdp.choose_action(left_prob)
            # print('move action', action)

            cur_state, rewards, done = Exp.step(cur_state, action)
            state_list.append(cur_state)
            reward_list.append(rewards)

        print('state trans: ', np.array(state_list))
        print('final reward: ', sum(reward_list))
        print('steps', len(reward_list))
        return torch.Tensor(action_list), torch.Tensor(np.array(state_list[:-1])), \
               torch.Tensor(mu_list), torch.Tensor(var_list)
    

    elif method_name == 'a2c_only':
        state_pos_max = 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Arc_pol = [3, 32, 64, 1]
        Arc_val = [3, 32, 64, 1]

        polnet = PolNet(Arc_pol, device)
        valnet = ValNet(Arc_val, device)
        
        polnet.train()
        valnet.train()
        
        agent = AgentA2C(polnet, valnet, device)

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
            mu_list.append(left_prob_mu)
            var_list.append(left_prob_var)
            value_list.append(val)
        
        # print('state trans: ', np.array(state_list))
        print('final reward: ', sum(reward_list))
        print('steps', len(reward_list))
        return torch.Tensor(action_list), torch.Tensor(np.array(state_list[:-1])), \
               torch.Tensor(mu_list), torch.Tensor(var_list)




if __name__ == "__main__":
    method = 'model_base'
    # method = 'a2c_only'

    action_batch, state_batch, mu_batch, var_batch = test_method(method)
    # print(state_batch)