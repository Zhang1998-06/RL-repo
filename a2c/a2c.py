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
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def get_action(self, states):
        states_v = torch.Tensor(states).to(self.device)

        ## Output real range
        act_range = [0.6, math.pi, 0.3, 0.4]
        act_mid_line = [0, 0, 0, 0.4]

        ## Distribution from network
        mu_v, var_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()

        ## Output distribution
        foot_mu = act_range[0]*mu[0:2]+act_mid_line[0]
        foot_std = act_range[0]*sigma[0:2]
        hd_mu = act_range[1]*mu[2]+act_mid_line[1]
        hd_std = act_range[1]*sigma[2]
        pos_mu = act_range[2]*mu[3:5]+act_mid_line[2]
        pos_std = act_range[2]*sigma[3:5]
        vel_mu = act_range[3]*mu[5:7]+act_mid_line[3]
        vel_std = act_range[3]*sigma[5:7]

        # Sampling
        fot_act = np.random.normal(foot_mu, foot_std)
        hd_act = np.random.normal(hd_mu, hd_std)
        pos_act = np.random.normal(pos_mu, pos_std)
        vel_act = np.random.normal(vel_mu, vel_std)

        # Control/actions
        fot_act = np.clip(fot_act, -act_range[0]+act_mid_line[0], act_range[0]+act_mid_line[0])
        hd_act = np.clip(hd_act, -act_range[1]+act_mid_line[1], act_range[1]+act_mid_line[1])
        pos_act = np.clip(pos_act, -act_range[2]+act_mid_line[2], act_range[2]+act_mid_line[2])
        vel_act = np.clip(vel_act, -act_range[3]+act_mid_line[3], act_range[3]+act_mid_line[3])

        # Concate
        action = np.concatenate([fot_act, [hd_act], pos_act, vel_act])
        return action
    
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    # args = parser.parse_args()
    states = torch.Tensor([1, 2, 1, 3, 2, 3, 1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Arc_pol = [29, 32, 64, 128, 32, 7]
    Arc_val = [29, 32, 64, 128, 32, 1]

    polnet = PolNet(Arc_pol, device)
    valnet = ValNet(Arc_val, device)
    
    polnet.train()
    valnet.train()
    
    agent = AgentA2C(polnet, device)
    act = agent.get_action(states)
    val = valnet(states)