#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


grid_h = 4
grid_w = 12

𝞮 = 0.1
𝜶 = 0.5
𝞬 = 1

up = 0
down = 1
left = 2
right = 3
actions = [[-1,0], [1,0], [0,-1], [0,1]]

S = [grid_h-1, 0]
G = [grid_h-1, grid_w-1]


# In[3]:


def move_agent(state, action):
    x, y = state
    if action == up:
        if x == 0:
            next_state = [x,y] 
        else:
            next_state = [x-1,y]
    elif action == left:
        if y == 0:
            next_state = [x,y] 
        else:
            next_state = [x,y-1]
    elif action == right:
        if y == 11:
            next_state = [x,y] 
        else:
            next_state = [x,y+1]
    elif action == down:
        if x == 3:
            next_state = [x,y] 
        else:
            next_state = [x+1,y]
    reward = -1
    if (action == down and x == 2 and 1 <= y <= 10) or (action == right and state == S):
        reward = -100
        next_state = S
    return next_state, reward


# In[4]:


def choose_action(s, Q):
    i,j = s
    if np.random.binomial(1, 𝞮):
        return np.random.choice([0,1,2,3])
    else:
        values_ = Q[i,j,:]
        return np.random.choice([a for a, v in enumerate(values_) if v == np.max(values_)])


# In[5]:


def sarsa(Q, 𝜶):
    s = S
    rewards = 0.0
    while s != G:
        i,j = s
        A = choose_action(s, Q) #Choose A from S using policy derived from Q
        S_, R = move_agent(s, A)
        i_, j_ = S_
        A_ = choose_action(S_, Q)
        rewards += R
        Q[i,j, A] += 𝜶 * (R + 𝞬*Q[i_, j_ , A_] - Q[i,j, A])
        s = S_
    return rewards


# In[6]:


def q_learning(Q, 𝜶):
    s = S
    rewards = 0.0
    while s != G:
        i,j = s
        A = choose_action(s, Q) #Choose A from S using policy derived from Q
        S_, R = move_agent(s, A)
        i_, j_ = S_
        rewards += R
        Q[i,j, A] += 𝜶 * (R + 𝞬*np.max(Q[i_, j_ , :]) - Q[i,j, A])
        s = S_
    return rewards


# In[7]:


def simulate():
    episodes = 500
    runs = 500
    rewards_sarsa = np.array([0]*episodes)
    rewards_ql = np.array([0]*episodes)
    for i in range(runs):
        Q_sarsa = np.zeros((grid_h, grid_w, 4))
        Q_ql = np.zeros((grid_h, grid_w, 4))
        for j in range(episodes):
            rewards_sarsa[j] += sarsa(Q_sarsa, 0.5)
            rewards_ql[j] += q_learning(Q_ql, 0.5)
    plt.plot(rewards_sarsa/runs, label='Sarsa')
    plt.plot(rewards_ql/runs, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('cliff.png')
    plt.close()


# In[8]:


simulate()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




