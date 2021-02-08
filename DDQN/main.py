import copy
from turtle import pd

import numpy as np
import pandas as pandas
import torch
import random
from matplotlib import pylab as plt
import gym
from collections import deque
import Box2D

env = gym.make('LunarLander-v2')
env.reset()
#-------------------------------------------------------------------------------------------------

def discretize(val,bounds,n_states):
    if val <= bounds[0]:
        discrete_val = 0
    elif val >= bounds[1]:
        discrete_val = n_states-1
    else:
        discrete_val = int(round((n_states-1)*((val-bounds[0])/(bounds[1]-bounds[0]))))
    return discrete_val

def discretize_state(vals,s_bounds,n_s):
    discrete_vals = []
    for i in range(len(n_s)):
        discrete_vals.append(discretize(vals[i],s_bounds[i],n_s[i]))
    return np.array(discrete_vals,dtype=np.int)

#-------------------------------------------------------------------------------------------------

file1 = open("Results.txt", "a")
lowest_losses = []
average_rewards = []
losses_avg_avg = []
rewards_avg = []
tests = 10

for test_number in range(tests):
    print(test_number)
    #-------------------------------------------------------------------------------------------------
    l1 = 8
    l2 = 256
    l3 = 4

    model = torch.nn.Sequential(
     torch.nn.Linear(l1, l2),
     torch.nn.ReLU(),
     torch.nn.Linear(l2, l3)
    )

    model2 = copy.deepcopy(model)
    model2.load_state_dict((model.state_dict()))
    sync_freq = 4

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gamma = 0.99
    epsilon = 1.0
    epochs = 1000
    memory = 50000
    batch = 64

    losses = []
    losses_avg = []
    losses_avg_counter = 0
    rewards = []
    rewards_avg_counter = 0
    replay = deque(maxlen=memory)
    j = 0

    #-------------------------------------------------------------------------------------------------
    for i in range(epochs):
        # Reset środowiska
        state = env.reset()
        sum_reward = 0
        while(True):
            j += 1

            # Wyliczenie wartości funkcji Q
            state_ = torch.from_numpy(state).float()
            qval = model(state_)
            qval_ = qval.data.numpy()

            # Eksploracja środowiska
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)

            # Wykonanie wybranej akcji
            new_state, reward, done, info = env.step(action_)
            sum_reward += reward

            # Zapisanie danych do pamięci
            exp = (state, action_, reward, new_state, done)
            replay.append(exp)

            state = new_state

            # Uczenie i aktualizacja wartości akcji
            if len(replay) > batch:
                minibatch = random.sample(replay, batch)
                state1_batch = torch.from_numpy(np.vstack([exp[0] for exp in minibatch])).float()
                action_batch = torch.from_numpy(np.vstack([exp[1] for exp in minibatch])).long()
                reward_batch = torch.from_numpy(np.vstack([exp[2] for exp in minibatch])).float()
                state2_batch = torch.from_numpy(np.vstack([exp[3] for exp in minibatch])).float()
                done_batch = torch.from_numpy(np.vstack([exp[4] for exp in minibatch])).float()

                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model2(state2_batch)

                Q1_argmax = torch.max(Q1, dim=1)[1]
                Q2_argmax = Q2.gather(dim=1, index=Q1_argmax.unsqueeze(dim=1))

                Y = reward_batch + gamma * ((1 - done_batch) * Q2_argmax)
                X = Q1.gather(1, action_batch)
                loss = loss_fn(X, Y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())

            if done == True:
                break

        if len(losses) != 0:
            losses_avg.append(np.average(losses))
            if len(losses_avg_avg) <= losses_avg_counter:
                losses_avg_avg.append(losses_avg[losses_avg_counter])
            else:
                losses_avg_avg[losses_avg_counter] += losses_avg[losses_avg_counter]
            losses_avg_counter += 1
            losses = []

        rewards.append(sum_reward)

        if len(rewards_avg) <= rewards_avg_counter:
            rewards_avg.append(rewards[rewards_avg_counter])
        else:
            rewards_avg[rewards_avg_counter] += rewards[rewards_avg_counter]
        rewards_avg_counter += 1

    plt.plot(losses_avg, 'r')
    plt.plot(pandas.Series(losses_avg).rolling(50).mean(), 'b')
    plt.savefig(str(test_number) + ' Loss')
    plt.clf()

    plt.plot(rewards)
    plt.plot(pandas.Series(rewards).rolling(50).mean(), 'r')
    plt.savefig(str(test_number) + ' Rewards')
    plt.clf()

    file1.write(str(test_number) + ": \n")
    file1.write("Lowest Loss: " + str(min(losses_avg)) + '\n')
    lowest_losses.append(min(losses_avg))

   # input()

    rewards.clear()
    #-------------------------------------------------------------------------------------------------
    for i in range(50):
        state = env.reset()
        sum_reward = 0
        #state = discretize_state(state, s_bounds, n_s)

        while(True):

            #if i % 10 == 0:
               # env.render()

            qval = model(torch.from_numpy(state).float())
            qval_ = qval.data.numpy()

            action_ = np.argmax(qval_)

            state, reward, done, info = env.step(action_)
            #state = discretize_state(state, s_bounds, n_s)
            sum_reward += reward

            if done == True:
                break

        rewards.append(sum_reward)

    file1.write("Average rewards: " + str(np.average(rewards)) + '\n' + '\n')
    average_rewards.append(np.average(rewards))

    #rewards.append(0)
    #plt.plot(rewards)
    #plt.show()

for x in range(len(losses_avg_avg)):
    losses_avg_avg[x] = losses_avg_avg[x]/tests

for x in range(len(rewards_avg)):
    rewards_avg[x] = rewards_avg[x] / tests

plt.plot(losses_avg_avg, 'r')
plt.plot(pandas.Series(losses_avg_avg).rolling(50).mean(), 'b')
plt.savefig('Loss_avg')
plt.clf()

plt.plot(rewards_avg, 'b')
plt.plot(pandas.Series(rewards_avg).rolling(50).mean(), 'r')
plt.savefig('Rewards_avg')
plt.clf()

file1.write('\n' + '\n' + '\n' + "Average lowest loss in general: " + str(np.average(lowest_losses)))
file1.write('\n' + "Average rewards in general: " + str(np.average(average_rewards)))
file1.close()
env.close()