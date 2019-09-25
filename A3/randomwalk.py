#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


values = np.zeros(7)
values[1:6] = 0.5
values[6] = 1

true_values = np.zeros(7)
for i in range(1,6):
    true_values[i] = i/6.
true_values[6] = 1

left = 0
right = 1


# In[3]:


def TD(values, 𝜶):
    new_state = 3 #Start from state C
    path = [new_state]
    rewards = [0]
    while True:
        state = new_state
        if np.random.binomial(1, 0.5): #Move to left or right with equal probability
            new_state += 1
        else:
            new_state -= 1
        reward = 0
        path.append(new_state)
        values[state] += 𝜶 * (reward + values[new_state] - values[state]) #v(s) = v[s] + 𝜶(r + v[s'] - v[s])
        if new_state == 6 or new_state == 0: #Terminate on reaching terminal states
            break
        rewards.append(reward)
    return path, rewards


# In[4]:


def MC(values, 𝜶):
    state = 3 #Start from state C
    path = [state]
    while True:
        if np.random.binomial(1, 0.5): #Move to left or right with equal probability
            state += 1
        else:
            state -= 1
        path.append(state)
        #No individual rewards for each step. Returns expected only at the end of episode
        if state == 6: #Returns on reaching the right terminal state is 1
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0 #Returns on reaching the right terminal state is 0
            break
    for s_ in path[:len(path)]:
        values[s_] += 𝜶 * (returns - values[s_]) #v[s] = v[s] + 𝜶*(G - v[s])
    return path, [returns] * (len(path) - 1)


# In[8]:


def compute_V():
    episodes = [0,1,10,100]
    current_v = np.copy(values)
    for i in range(101):
        if i in episodes:
            if i == 1:
                plt.plot(current_v, label=str(i) + ' episode')
            else:
                plt.plot(current_v, label=str(i) + ' episodes')
        TD(current_v, 0.1)
    plt.plot(true_values, label='true')
    plt.xlim([1,5])
    plt.xlabel('State')
    plt.ylabel('Estimated value')
    plt.legend()
    plt.savefig('1.png')
    plt.close()


# In[9]:


def rms_error():
    td_𝜶 = [0.05, 0.1, 0.15]
    mc_𝜶 = [0.01, 0.02, 0.03, 0.04]
    episodes = 101
    runs = 100
    for 𝜶 in td_𝜶:
        errors = np.zeros(episodes)
        for i in range(runs):
            current_v = np.copy(values)
            for j in range(episodes):
                errors[j] += (np.sum((true_values-current_v)**2)/5.0)**0.5
                TD(current_v, 𝜶)
        plt.plot(errors/runs, linestyle='solid', label='TD' + ', ' +  r'$\alpha$' + ' = %.02f' % (𝜶))        
    for 𝜶 in mc_𝜶:
        errors = np.zeros(episodes)
        for i in range(runs):
            current_v = np.copy(values)
            for j in range(episodes):
                errors[j] += (np.sum((true_values-current_v)**2)/5.0)**0.5
                MC(current_v, 𝜶)
        plt.plot(errors/runs, linestyle='dashdot', label='MC' + ', ' +  r'$\alpha$' + ' = %.02f' % (𝜶))     
    plt.xlabel('episodes')
    plt.ylabel('RMS')
    plt.legend()
    plt.savefig('2.png')
    plt.close()


# In[10]:


compute_V()
rms_error()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




