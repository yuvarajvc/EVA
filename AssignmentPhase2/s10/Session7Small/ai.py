# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating the architecture of the Neural Network

#CNN for getting observation space

# class ObsSpaceNetwork(nn.Module):
#     def __init__(self, ALPHA=0.01):
#         super(ObsSpaceNetwork, self).__init__()
#         #self.conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=1)
#         self.conv1 = nn.Conv2d(1, 32, 6, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
#         self.conv3 = nn.Conv2d(64, 128, 3)
#         #self.fc1 = nn.Linear(128*23*16, 512)
#         self.fc1 = nn.Linear(128*16*16, 512)
#         self.fc2 = nn.Linear(512, 6)
#         #self.optimizer = optim.SGD(self.parameters(), lr=self.ALPHA, momentum=0.9)
#         self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
#         self.loss = nn.MSELoss()
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.to(self.device)
#         self.ALPHA = 0.01

#     def sceneforward(self, sceneparam):
#         #observation = torch.from_numpy(observation)
#         sceneparam = torch.Tensor(sceneparam).to(self.device)
#         #observation = observation.view(-1, 3, 210, 160).to(self.device)
#         sceneparam = sceneparam.view(-1, 1, 80, 80)
#         sceneparam = F.relu(self.conv1(sceneparam))
#         sceneparam = F.relu(self.conv2(sceneparam))
#         sceneparam = F.relu(self.conv3(sceneparam))
#         #observation = observation.view(-1, 128*23*16).to(self.device)
#         sceneparam = sceneparam.view(-1, 128*16*16)
#         sceneparam = F.relu(self.fc1(sceneparam))
#         observationspace = self.fc2(sceneparam)
#         return observationspace

""" class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        print("NetworkInit")
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):
        print("NetworkForward")
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay """

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


class Actor(nn.Module):

  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.max_action = max_action
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 1)
    #self.fc3 = nn.Linear(3, 1)    
              

  def forward(self, x):
    print("actor forward")
    x = x.view(-1, 1, 28, 28)
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    #x = F.relu(self.fc2(x))
    # = F.dropout(x)
    x = self.max_action * torch.tanh(self.fc2(x))
    return  x
    
    #F.log_softmax(x)
class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 100)
    self.fc2 = nn.Linear(101, 50)
    self.fc3 = nn.Linear(50, 1)
    # Defining the second Critic neural network
    self.conv3 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv4 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv4_drop = nn.Dropout2d()
    self.fc4 = nn.Linear(320, 100)
    self.fc5 = nn.Linear(101, 50)
    self.fc6 = nn.Linear(50, 1)

  def forward(self, x, u):
    #xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    print("critic forward")
    #u.reshape(100, 1)
    x1 = x.view(-1, 1, 28, 28)
    x1 = F.relu(F.max_pool2d(self.conv1(x1), 2))
    x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
    x1 = x1.view(-1, 320)
    x1 = F.relu(self.fc1(x1))
    xu1 = torch.cat([x1, u], 1)
    x1 = F.relu(self.fc2(xu1))
    x1 = F.relu(self.fc3(x1))
    # Forward-Propagation on the second Critic Neural Network
    x2 = x.view(-1, 1, 28, 28)
    x2 = F.relu(F.max_pool2d(self.conv1(x2), 2))
    x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x2)), 2))
    x2 = x2.view(-1, 320)
    x2 = F.relu(self.fc1(x2))
    xu2 = torch.cat([x2, u], 1)
    x2 = F.relu(self.fc2(xu2))
    x2 = F.relu(self.fc3(x2))
    return x1, x2

  def Q1(self, x, u):
    x1 = x.view(-1, 1, 28, 28)
    x1 = F.relu(F.max_pool2d(self.conv1(x1), 2))
    x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
    x1 = x1.view(-1, 320)
    x1 = F.relu(self.fc1(x1))
    print("Q1")
    xu1 = torch.cat([x1, u], 1)
    x1 = F.relu(self.fc2(xu1))
    x1 = F.relu(self.fc3(x1))
    return x1

# Selecting the device (CPU or GPU)


# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).unsqueeze(1).to(device)
      next_state = torch.Tensor(batch_next_states).unsqueeze(1).to(device)
      #print(next_state.shape)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      #print(next_action.shape)
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions.reshape(batch_size, 1)).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

      #noise = torch.Tensor(batch_actions.reshape(batch_size, 1)).data.normal_(0, policy_noise).to(device)
      #noise = noise.clamp(-noise_clip, noise_clip)
      #print(noise.shape)
      
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action.reshape(batch_size, 1))
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        print("helloman")
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
# Implementing Deep Q Learning


