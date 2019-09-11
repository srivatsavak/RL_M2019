#!/usr/bin/env python
# coding: utf-8

# In[133]:


import numpy as np


# In[134]:


𝞬 = 0.9
θ = 1e-4
actions = [[0,-1],[-1,0],[0,1],[1,0]] #Left, Up, Right, Down respectively
action_prob = 1/len(actions)


# In[135]:


def getNextState(state, action):
    if state == [0,1]:
        return [4,1], 10
    elif state == [0,3]:
        return [2,3], 5
    else:
        x,y = (np.array(state)+np.array(action)).tolist() #change to next state
        if x<0 or x>=5 or y<0 or y>=5:
            reward = -1.0 #with a negative reward
            x,y = state
        else: #no penalty or reward if it goes off the grid
            reward = 0
        return (x,y), reward


# In[136]:


def myFunc():
    val = np.zeros((5,5))
    while True:
        val_ = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                for a in actions: #for each action in a set of actions 
                    (i_, j_), reward = getNextState([i, j], a)
                    val_[i, j] += action_prob*(reward+(𝞬*val[i_, j_])) #new_val = Σp(s',r|s,a)[r+𝞬*old_val(s')]
        Δ = abs(val - val_).max()
        if Δ < θ: #run until the following condition is met
            print(np.round(val_, decimals=1))
            break
        val = val_


# In[137]:


myFunc()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




