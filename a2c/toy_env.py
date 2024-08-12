import numpy as np
from torch import nn

class Toy_env():
    def __init__(self):
        self.list = [0, 1, 2, 3, 4]
        self.init_pos = self.list[1]
        self.goal = self.list[0]
        self.mid_pt = self.list[4]
        self.reach_mid_pt = 0
        self.reach_goal = 0
    
    def reset(self):
        cur_pos = self.list[1]
        self.reach_mid_pt = 0
        self.reach_goal = 0
        state = np.array([cur_pos, self.reach_mid_pt, self.reach_goal])
        return state
    
    def step(self, state, act):
        cur_pos = self.list[state[0]]
        cur_pos = self.list[cur_pos+act]
        rewars = -0.01
        if self.reach_mid_pt == 0:
            if cur_pos == self.mid_pt:
                self.reach_mid_pt = 1
            
            if cur_pos == self.goal:
                rewars = 0.3
                self.reach_goal = 1
        else:
            if cur_pos == self.goal:
                rewars = 1
                self.reach_goal = 1
        state = np.array([cur_pos, self.reach_mid_pt, self.reach_goal])

        return state, rewars, self.reach_goal
        

if __name__ == "__main__":
    Exp = Toy_env()
    state = Exp.reset()
    action = 1
    state_1, rewards, done = Exp.step(state, action)
    print(state)
    print(state_1)
    print(rewards)
    print(done)