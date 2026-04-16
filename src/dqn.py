import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer — stores past transitions
    so the agent can learn from them in random batches.
    Breaks correlation between consecutive experiences.
    """

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    Architecture: input(6) -> 128 -> 128 -> output(4)
    """

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    Uses PyTorch for stable gradient computation.
    """

    def __init__(self, state_size, action_size,
                 lr=0.0005, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update=200):

        self.state_size    = state_size
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps         = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online and target networks
        self.online = QNetwork(state_size, action_size).to(self.device)
        self.target = QNetwork(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()  # Huber loss — more stable than MSE

        self.memory = ReplayBuffer(capacity=50000)

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        with torch.no_grad():
            state_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online(state_t)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current Q values for taken actions
        q_current = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values — Bellman equation
        with torch.no_grad():
            q_next   = self.target(next_states).max(1)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        # Sync target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.online.state_dict())