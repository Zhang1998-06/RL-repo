from gym import spaces
import random
import numpy as np
import random
from collections import defaultdict
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mdp import StochasticMDPEnv
from tqdm import tqdm

class RuleBasedController:
    def __init__(self):
        self.visited_six = False

    def select_action(self, current_state):
        ruleprob=self.action_distribution(current_state)
        
        # Take action based on the probability distribution
        if random.random() < ruleprob[1]:
            return 1  # Move right
        else:
            return 0  # Move left
        
    def action_distribution(self,current_state):
        if current_state == 5:
            self.visited_six = True       

        if not self.visited_six:
            # If state 6 has not been visited
            ruleprob = [0.2,0.8]
        else:
            # If state 6 has been visited
            ruleprob = [0.8,0.2]   
        return ruleprob
         
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        
        self.device = device

    def take_action(self, state):
        state = self.one_hot_encode(state)
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        #print(probs)
        action_dist = torch.distributions.Categorical(probs)
        #print(action_dist)
        action = action_dist.sample()
        #print(action)
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor([self.one_hot_encode(s) for s in transition_dict['states']], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor([self.one_hot_encode(s) for s in transition_dict['next_states']], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states) # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def one_hot_encode(self, state):
        one_hot = np.zeros(self.actor.fc1.in_features)
        one_hot[state] = 1
        return one_hot
    
class PCYAC:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device


    def take_action(self, state):
        state = self.one_hot_encode(state)
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), probs

    def update(self, transition_dict):
        states = torch.tensor([self.one_hot_encode(s) for s in transition_dict['states']], dtype=torch.float).to(self.device)
        
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor([self.one_hot_encode(s) for s in transition_dict['next_states']], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states) # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def one_hot_encode(self, state):
        one_hot = np.zeros(self.actor.fc1.in_features)
        one_hot[state] = 1
        return one_hot



def train_on_policy_MPCAC(env, agent, num_episodes,alpha):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                MPC= RuleBasedController()
                state = env.reset()
                done = False
                while not done:
                    action, probs = agent.take_action(state)
                    #MPC_action = MPC.action_distribution(state)
                    next_state, reward, done, _ = env.step(action)

                    # Compute rule-based probabilities
                    with torch.no_grad():
                        rule_probs = torch.tensor(MPC.action_distribution(state), dtype=torch.float).to(device)

                    # Compute KL divergence
                    probs_tensor = torch.tensor(probs, dtype=torch.float).to(device)
                    kl_div = F.kl_div(probs_tensor.log(), rule_probs, reduction='batchmean')
                    kl_div = torch.clamp(kl_div, max=1.0)

                    # Modify reward with KL divergence
                    new_reward = reward-alpha * kl_div.item()

                    # Normalize reward
                    reward_min = 0 - alpha
                    reward_max = 1
                    new_reward = (new_reward - reward_min) / (reward_max - reward_min)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(new_reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
        # Save return list to CSV
    pd.DataFrame(return_list).to_csv('return_list_MPCAC.csv', index=False)
    return return_list
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    #MPCaction=MPC.action_distribution(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    #transition_dict['MPCaction'].append(MPCaction)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                 # Save return list to CSV
    pd.DataFrame(return_list).to_csv('return_list_agent.csv', index=False)
    return return_list



def evaluate_rb_controller(env,  num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                controller= RuleBasedController()
                state = env.reset()
                episode_return = 0
                done = False
                while not done:
                    action = controller.select_action(state)
                    state, reward, done, _ = env.step(action)
                    episode_return += reward
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                # Save return list to CSV
    pd.DataFrame(return_list).to_csv('return_list_rb.csv', index=False)
    return return_list



actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 5000
hidden_dim = 64
gamma = 1
alpha=0.01
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = StochasticMDPEnv()
state_dim = env.nS  # Using the number of states as the dimension for one-hot encoding
action_dim = env.nA
episodes_list = list(range(num_episodes))


# Train Actor-Critic agent baseline1
ac_agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
ac_return_list = train_on_policy_agent(env, ac_agent, num_episodes)



# Train Actor-Critic agent with reward inspired policy 
a2cPCY_agent = PCYAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,  device)
a2cPCY_return_list = train_on_policy_MPCAC(env, a2cPCY_agent,num_episodes,alpha)



# Evaluate Rule-Based Controller

rb_return_list = evaluate_rb_controller(env,  num_episodes)




def plot_mean_rewards(ac_returns, a2c_returns, rule_based_returns, mean_number,smoothing_window):
    episodes_list = range(1, len(ac_returns) + 1)

    # Calculate mean reward for every mean_number episodes
    ac_returns_series = pd.Series(ac_returns).rolling(window=smoothing_window, min_periods=1).mean()
    a2c_returns_series = pd.Series(a2c_returns).rolling(window=smoothing_window, min_periods=1).mean()
    rule_based_series = pd.Series(rule_based_returns).rolling(window=smoothing_window, min_periods=1).mean()
    
    ac_mean_reward = np.mean(ac_returns_series.values.reshape(-1, mean_number), axis=1)
    a2c_mean_reward = np.mean(a2c_returns_series.values.reshape(-1, mean_number), axis=1)
    rule_based_mean_reward = np.mean(rule_based_series.values.reshape(-1, mean_number), axis=1)
    
    mean_axis = np.arange(mean_number, len(ac_returns) + 1, mean_number)

    # Plot the mean rewards
    plt.figure(figsize=(10, 5))
    plt.plot(mean_axis, ac_mean_reward, label='Actor-Critic',  linewidth=1)
    plt.plot(mean_axis, a2c_mean_reward, label='A2C(with suboptimal policy)', linewidth=1)
    plt.plot(mean_axis, rule_based_mean_reward, label='Rule-Based suboptimal policy', linewidth=1)

    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('Mean Returns Over Episodes')
    plt.legend()
    plt.show()
'''
# Example usage with random data (replace with your actual data)
ac_returns = ac_return_list  # Replace with your actual return data for Actor-Critic
a2c_returns = a2cPCY_return_list   # Replace with your actual return data for A2CPCY
rule_based_returns = rb_return_list  # Replace with your actual return data for Rule-Based
'''
# Load return lists from CSV
if __name__ == "__main__":
    ac_returns = pd.read_csv('return_list_agent.csv').squeeze().tolist()
    a2c_returns = pd.read_csv('return_list_MPCAC.csv').squeeze().tolist()
    rule_based_returns = pd.read_csv('return_list_rb.csv').squeeze().tolist()
    plot_mean_rewards(ac_returns, a2c_returns, rule_based_returns, mean_number=50,smoothing_window=50)
