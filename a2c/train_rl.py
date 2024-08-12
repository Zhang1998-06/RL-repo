import os
import math
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from run_env_rl import run_toy
from a2c_toy import PolNet, ValNet, AgentA2C


LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-2
RUN_TIMES = 1
GAMMA = 0.97

if __name__ == "__main__":
    load_path = 'rl_model/'
    method = 'a2c_only'

    # RL model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Arc_pol = [3, 32, 64, 1]
    Arc_val = [3, 32, 64, 1]

    polnet = PolNet(Arc_pol, device)
    valnet = ValNet(Arc_val, device)

    if os.path.exists(load_path+'rl_polnet.pth'):
        state_dict_pol = torch.load(load_path+'rl_polnet.pth')
        polnet.load_state_dict(state_dict_pol)
    if os.path.exists(load_path+'rl_valnet.pth'):
        state_dict_val = torch.load(load_path+'rl_valnet.pth')
        valnet.load_state_dict(state_dict_val)

    optimizer_pol = optim.Adam(polnet.parameters(), lr = LEARNING_RATE)
    optimizer_val = optim.Adam(valnet.parameters(), lr = LEARNING_RATE)

    # Data save
    ten_avg_reward = np.array([])
    ten_avg_epoch = np.array([])
    rewards = np.array([])
    epoch = np.array([])

    for step_idx in range(RUN_TIMES):
        polnet.train()
        valnet.train()
        agent = AgentA2C(polnet, valnet, device)
        act_v, state_v, mu_v, \
        var_v, ref_val_v, final_reward = run_toy(method, agent, GAMMA)

        optimizer_pol.zero_grad()
        optimizer_val.zero_grad()
        val_batch = agent.get_value(state_v)

        action_batch = torch.Tensor(act_v).to(device)
        ref_val_batch = torch.Tensor(ref_val_v).to(device)
        # ref_val_batch = torch.unsqueeze(ref_val_batch, 1)
        val_batch = torch.squeeze(val_batch, 1)
        mu_batch = torch.Tensor(mu_v).to(device)
        var_batch = torch.Tensor(var_v).to(device)
        
        print('action shapt', action_batch.shape)
        print('ref_val shapt', ref_val_batch.shape)
        print('val shapt', val_batch.shape)
        print('action shapt', action_batch.shape)
        

        # Value loss
        loss_value_v = F.mse_loss(val_batch, ref_val_batch)
        print('mse loss: ', loss_value_v)

        # Policy loss
        adv_v = ref_val_batch - val_batch.detach()
        log_prob_v = adv_v*agent.cal_logprob(mu_batch, var_batch, action_batch)
        loss_pol_v = -log_prob_v.mean()
        print('pol_loss: ', loss_pol_v)

        # Entropy loss
        entro_loss_v = ENTROPY_BETA*(-(torch.log(2*math.pi*var_batch)+1)/2).mean()
        print('entropy_loss: ', entro_loss_v)

        loss_val = loss_value_v
        loss_val.backward()
        optimizer_val.step()

        loss_act = loss_pol_v + entro_loss_v
        loss_act.backward()
        optimizer_pol.step()
        
        rewards = np.append(rewards, final_reward)
        epoch = np.append(epoch, step_idx)

        if np.mod(step_idx, 10) == 0:
            ten_avg_epoch = np.append(ten_avg_epoch, step_idx)
            avg_rewd = sum(rewards)/len(rewards)
            ten_avg_reward = np.append(ten_avg_reward, avg_rewd)
            rewards = np.array([])

            res_state_dict_pol = polnet.state_dict()
            res_state_dict_val = valnet.state_dict()

            torch.save(res_state_dict_pol, load_path+'rl_polnet.pth')
            torch.save(res_state_dict_val, load_path+'rl_valnet.pth')





