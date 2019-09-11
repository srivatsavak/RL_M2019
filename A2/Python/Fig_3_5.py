#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np


# In[33]:


𝞬 = 0.9
θ = 1e-4
actions = [[0,-1],[-1,0],[0,1],[1,0]] #Left, Up, Right, Down respectively
action_prob = 1/len(actions)


# In[34]:


def getNextState(state, action):
    if state == [0,1]:
        return [4,1], 10
    elif state == [0,3]:
        return [2,3], 5
    else:
        x,y = (np.array(state)+np.array(action)).tolist() #Change to next state
        if x<0 or x>=5 or y<0 or y>=5:
            reward = -1.0 #with a negative reward
            x,y = state
        else: #no penalty or reward if it goes off the grid
            reward = 0
        return (x,y), reward


# In[35]:


def myFunc():
    val = np.zeros((5,5))
    while True:
        val_ = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                values = []
                for a in actions: #for each action in a set of Actions
                    (i_, j_), reward = getNextState([i, j], np.array(a)) #new state and reward
                    values.append(reward+(𝞬*val[i_, j_]))
                val_[i, j] = np.max(values) #new_val = max Σp(s',r|s,a)[r+𝞬*old_val(s')]
        Δ = abs(val - val_).max()
        if Δ < θ: #run until the following condition is met
            print(np.round(val_, decimals=1))
            break
        val = val_


# In[36]:


myFunc()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




