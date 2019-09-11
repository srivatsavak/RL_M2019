#!/usr/bin/env python
# coding: utf-8

# In[140]:


import numpy as np


# In[141]:


𝞬 = 1 #undiscounted
θ = 1e-4
actions = [[0,-1],[-1,0],[0,1],[1,0]] #Left, Up, Right, Down respectively
action_prob = 1/len(actions)


# In[142]:


def getNextState(state, action):
    x, y = state
    if (x == 0 and y == 0) or (x == 3 and y == 3):
        return (x,y), 0
    else:
        x,y = (np.array(state)+np.array(action)).tolist()
        if x<0 or x>=4 or y<0 or y>=4:      
            x,y = state
        return (x,y), -1.0


# In[143]:


def policyIter():
    val_ = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0]).reshape(4,4)
    #val_ = np.zeros((4,4))
    iteration = 0
    while True:
        val = val_
        temp = val.copy()
        for i in range(4):
            for j in range(4):
                value = 0
                for a in actions:
                    (i_, j_), r = getNextState([i, j], a)
                    value += action_prob*(r + 𝞬*val[i_, j_]) #new_val = Σp(s',r|s,a)[r+𝞬*old_val(s')]
                val_[i, j] = value
        Δ = abs(temp - val_).max()
        if Δ < θ: #run until the following condition is met
            print(np.round(val_, decimals=1))
            break
        iteration += 1


# In[144]:


policyIter()


# In[145]:


def valIter(in_place=True):
    val_ = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0]).reshape(4,4)
    #val_ = np.zeros((4,4))
    iteration = 0
    while True:
        val = val_
        temp = val.copy()
        for i in range(4):
            for j in range(4):
                values = []
                for a in actions:
                    (i_, j_), r = getNextState([i, j], np.array(a)) #new state and reward
                    values.append(action_prob * (r + 𝞬 * val[i_, j_])) 
                val_[i, j] = np.max(values) #new_val = max Σp(s',r|s,a)[r+𝞬*old_val(s')]
        Δ = abs(temp - val_).max()
        print (val_)
        print()
        if Δ < θ: #run until the following condition is met
            print(np.round(val_, decimals=1))
            break
        iteration += 1


# In[146]:


valIter()


# In[ ]:





# In[ ]:





# In[ ]:




