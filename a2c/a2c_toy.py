import torch
import math
import argparse
from torch import nn
from torch.nn import functional as F
import numpy as np

class PolNet(nn.Module):
    def __init__(self, Arc, device):
        ## Arc don't need the output dimention
        super(PolNet, self).__init__()
        self.Arc = Arc
        self.device = device
        self.mu = nn.Sequential(nn.Tanh())
        self.var = nn.Sequential(nn.Softplus())
        self.model = self.create_model().to(self.device)

    def create_model(self):
        layers = []
        for i in range(len(self.Arc)-2):
            layers.append(nn.Linear(self.Arc[i], self.Arc[i+1], bias = True))
            layers.append(nn.ReLU())
            if i < len(self.Arc) - 3:
                layers.append(nn.Dropout(p=0.1))
            
        layers.append(nn.Linear(self.Arc[-2], self.Arc[-1], bias = True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return self.mu(out), self.var(out)


class ValNet(nn.Module):
    def __init__(self, Arc, device):
        super(ValNet, self).__init__()
        self.Arc = Arc
        self.device = device
        self.model = self.create_model().to(self.device)

    def create_model(self):
        layers = []
        for i in range(len(self.Arc)-2):
            layers.append(nn.Linear(self.Arc[i], self.Arc[i+1], bias = True))
            layers.append(nn.ReLU())
            if i < len(self.Arc) - 3:
                layers.append(nn.Dropout(p=0.1))
            
        layers.append(nn.Linear(self.Arc[-2], self.Arc[-1], bias = True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return out
    

class AgentA2C():
    def __init__(self, polnet, valnet, device):
        self.polnet = polnet
        self.valnet = valnet
        self.device = device

    def get_action(self, states, need_act = True):
        states_v = torch.Tensor(states).to(self.device)

        ## Distribution from network
        out_mu, out_var = self.polnet(states_v)
        if states[0] == 1:
            out_mu = torch.Tensor([1])
            out_var = torch.Tensor([0.001])
        sigma_v = torch.sqrt(out_var)
        
        ## Output distribution
        if need_act:
            act_mu = out_mu.data.cpu().numpy()
            act_std = sigma_v.data.cpu().numpy()

            # Sampling
            left_prob_act = np.random.normal(act_mu, act_std)
            if states[0] == 1:
                left_prob_act = 1
            # actions/policy
            left_prob_act = np.clip(left_prob_act, 0, 1)
            
  
            return left_prob_act, out_mu, out_var         # output type: np array
        else:
            return out_mu, out_var                        # output type: torch tensor
    
    def get_value(self, states):
        states_v = torch.Tensor(states).to(self.device)

        # Calculate state value
        val = self.valnet(states_v)
        return val                      # output type: torch tensor

    def cal_logprob(self, mu_v, var_v, actions_v):
        p1 = -((mu_v - actions_v)**2/(2*var_v.clamp(min = 1e-3)))
        p2 = -torch.log(torch.sqrt(2*math.pi*var_v))
        return p1 + p2                  # output type: torch tensor
    
    def cal_klentro(self, mu_batch, var_batch, tar_mu, tar_var):
        sig_batch = torch.sqrt(var_batch)
        tar_sig = torch.sqrt(tar_var)
        ent_1 = torch.log(tar_sig/sig_batch)
        ent_2 = (var_batch+torch.pow((mu_batch-tar_mu),2))/(2*tar_var)
        return ent_1+ent_2-1/2
    

if __name__ == "__main__":
    state_pos_max = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # states = torch.Tensor([[1, 0, 0],
    #                        [2, 0, 0],
    #                        [3, 0, 0]]).to(device)

    states = torch.Tensor([1, 0, 0]).to(device)

    norm_states = states/torch.Tensor([state_pos_max, 1, 1])
    print(norm_states)

    Arc_pol = [3, 32, 64, 1]
    Arc_val = [3, 32, 64, 1]

    polnet = PolNet(Arc_pol, device)
    valnet = ValNet(Arc_val, device)
    
    polnet.train()
    valnet.train()
    
    agent = AgentA2C(polnet, valnet, device)
    act, left_prob_mu, left_prob_var = agent.get_action(norm_states)
    val = agent.get_value(norm_states)
    
    print('left_prob: ', act)
    print('left_prob_mu: ', left_prob_mu)
    print('left_prob_var: ', left_prob_var)
    print('state_values: ', val)
