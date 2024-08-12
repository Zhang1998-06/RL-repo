import numpy as np
from torch import nn
import random

class Model_MDP():
    def __init__(self, init_pos, goal_pos, mid_pos):
        self.start = init_pos
        self.goal = goal_pos
        self.mid = mid_pos

    
    def choose_prob(self, cur_state):
        reach_mid = cur_state[1]
        if reach_mid == True:
            self.left_prob_mu = 0.7
            self.sig = 0.05
        else:
            self.left_prob_mu = 0.3
            self.sig = 0.05

        if cur_state[0] == 4:
            self.left_prob_mu = 1
            self.sig = 0.0001

        left_prob = self.samp_left_prob()
        # rigt_prob = 1- left_prob
        # print('left_prob: ', left_prob)
        # print('right_prob: ', rigt_prob)
        return left_prob
    

    def samp_left_prob(self):
        left_prob = np.random.normal(self.left_prob_mu, self.sig)
        return left_prob
    

    def choose_action(self, left_prob):
        samp_rand = random.random()
        # print('samp_rand', samp_rand)
        if samp_rand <= left_prob:
            action = -1
        else:
            action = 1

        return action



if __name__ == "__main__":
    init_pos = 1
    goal_pos = 0
    mid_pos = 4
    mdp = Model_MDP(init_pos, goal_pos, mid_pos)

    cur_state = [3, 1, 0]
    left_prob = mdp.choose_prob(cur_state)
    print('left prob', left_prob)
    action = mdp.choose_action(left_prob)
    print('move action', action)