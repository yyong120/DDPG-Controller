import numpy as np
import torch
import torch.nn as nn
import torch.functional as Func
import torch.optim as optim
import torch.utils.data as Data
import sys
import os

from cstr_params import *

class GActionNoise():
    def __init__(self, mu, sigma, decay_factor):
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.decay_factor = decay_factor
    
    def __call__(self):
        x = np.random.normal(self.mu, self.sigma, self.mu.shape)
        return x
    
    def decay(self):
        self.sigma *= self.decay_factor


class ReplayBuffer():
    def __init__(self, max_size, input_size, action_size, max_reward):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.max_reward = max_reward
        
        self.state_memory = np.zeros((self.mem_size, input_size))
        self.action_memory = np.zeros((self.mem_size, action_size))
        self.reward_memory = np.zeros(self.mem_size)
        self.new_state_memory = np.zeros((self.mem_size, input_size))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)
    
    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_cntr % self.mem_size
        
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = new_state
        self.terminal_memory[idx] = done
        
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


class CriticFc(nn.Module):
    def __init__(self, c_lr, input_size, fc1_dims, fc2_dims, action_size, 
                 input_bds, action_bds, name, chkpt_dir='tmp/ddpg'):
        super(CriticFc, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_bds = np.array(input_bds)
        self.action_bds = np.array(action_bds)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg.pkl')

        # normalization layer
        self.nl = nn.Linear(self.input_size + self.action_size, self.input_size + self.action_size)
        state_action_bds = 1.0 / np.concatenate((self.input_bds, self.action_bds))
        state_action_bds = np.diag(state_action_bds)
        self.nl.weight = nn.Parameter(torch.tensor(state_action_bds))
        self.nl.bias = nn.Parameter(torch.zeros(self.input_size + self.action_size))
        for p in self.nl.parameters():
            p.requires_grad = False

        self.fc1 = nn.Linear(self.input_size + self.action_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        f3 = 0.01
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=c_lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, action):
        state_action = torch.cat((state, action), 1)
        state_action = self.nl(state_action)
        state_action_value = torch.relu(self.fc1(state_action))
        state_action_value = torch.relu(self.fc2(state_action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value
    
    def save_checkpoint(self):
        # print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorFc(nn.Module):
    def __init__(self, a_lr, input_size, fc1_dims, fc2_dims, action_size,
                 input_bds, action_bds, name, chkpt_dir='tmp/ddpg'):
        super(ActorFc, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_bds = np.array(input_bds)
        self.action_bds = np.array(action_bds)
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.pkl')

        # normalization layer
        self.nl = nn.Linear(self.input_size, self.input_size)
        state_bds = np.diag(1.0 / self.input_bds)
        self.nl.weight = nn.Parameter(torch.tensor(state_bds))
        self.nl.bias = nn.Parameter(torch.zeros(self.input_size))
        for p in self.nl.parameters():
            p.requires_grad = False

        self.fc1 = nn.Linear(self.input_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.u = nn.Linear(self.fc2_dims, self.action_size)
        f3 = 0.01
        nn.init.uniform_(self.u.weight.data, -f3, f3)
        nn.init.uniform_(self.u.bias.data, -f3, f3)

        # scaling layer
        self.sl = nn.Linear(self.action_size, self.action_size)
        action_bds = np.diag(self.action_bds)
        self.sl.weight = nn.Parameter(torch.tensor(action_bds))
        self.sl.bias = nn.Parameter(torch.zeros(self.action_size))
        for p in self.sl.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=a_lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        state = self.nl(state)
        action = torch.relu(self.fc1(state))
        action = torch.relu(self.fc2(action))
        action = torch.tanh(self.u(action))
        action = self.sl(action)

        return action

    def save_checkpoint(self):
        # print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))


class AgentFc():
    def __init__(self, a_lr, c_lr, input_size, action_size, input_bds, action_bds, max_reward,
                 action_noise_mu, action_noise_sigma, action_noise_decay, tau,
                 env, discount=0.99, max_size=10000, layer1_size=12,
                 layer2_size=8, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.discount = discount
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_size, action_size, max_reward)
        self.batch_size = batch_size

        self.critic = CriticFc(c_lr, input_size, layer1_size, layer2_size,
                               action_size, input_bds, action_bds, 'Critic', chkpt_dir).double()
        self.actor = ActorFc(a_lr, input_size, layer1_size, layer2_size,
                             action_size, input_bds, action_bds, 'Actor', chkpt_dir).double()
        self.target_critic = CriticFc(c_lr, input_size, layer1_size, layer2_size,
                                      action_size, input_bds, action_bds, 'TargetCritic', chkpt_dir).double()
        self.target_actor = ActorFc(a_lr, input_size, layer1_size, layer2_size,
                                    action_size, input_bds, action_bds, 'TargetActor', chkpt_dir).double()
        
        self.noise = GActionNoise(action_noise_mu, action_noise_sigma, action_noise_decay)

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation, add_noise):
        observation = torch.tensor(observation).double().to(self.actor.device)
        u = self.actor(observation).to(self.actor.device)
        if add_noise:
            u += torch.tensor(self.noise()).double().to(self.actor.device)
        
        return u.cpu().detach().numpy()
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def get_target_critic_value(self, reward, new_critic_value_tensor, done):
        '''
        'new_critic_value_tensor' is a tensor containing a single element,
        so use new_critic_value_tensor.item() to get its value

        if done == 0:
            not a terminal state
        if done == 1:
            exceed the range
        if done == 2:
            converge to 0
        '''
        ### argmax ###
        # if done == 0:
        #     return reward + self.discount * new_critic_value_tensor.item()
        # if done == 1:
        #     return reward + 0.0
        # return reward + self.memory.max_reward

        ### argmin ###
        if done == 0:
            return reward + self.discount * new_critic_value_tensor.item()
        if done == 2:
            return reward + 0.0
        return reward + self.memory.max_reward


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        state = torch.tensor(state).double().to(self.critic.device)
        action = torch.tensor(action).double().to(self.critic.device)
        reward = torch.tensor(reward).double().to(self.critic.device)
        new_state = torch.tensor(new_state).double().to(self.critic.device)

        new_action = self.target_actor(new_state)
        new_critic_value = self.target_critic(new_state, new_action)
        critic_value = self.critic(state, action)

        target_critic_value = list(map(self.get_target_critic_value, reward, new_critic_value, done))
        target_critic_value = torch.tensor(target_critic_value).to(self.critic.device)
        target_critic_value = target_critic_value.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        loss = nn.MSELoss()
        critic_loss = loss(target_critic_value, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        u = self.actor(state)
        self.actor.optimizer.zero_grad()
        # actor_loss = -self.critic(state, u) # argmax
        actor_loss = self.critic(state, u)  # argmin
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_dict = dict(critic_params)
        actor_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_dict:
            critic_dict[name] = tau * critic_dict[name].clone() + \
                                (1-tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_dict)
        
        for name in actor_dict:
            actor_dict[name] = tau * actor_dict[name].clone() + \
                                (1-tau) * target_actor_dict[name].clone()
        
        self.target_actor.load_state_dict(actor_dict)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
