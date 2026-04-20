import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .hdc_encoding import D


class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # จะใช้เป็น Q-value ต่อ action (per-action network)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim=D,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        device=None,
    ):
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim).to(self.device)
        self.target_net = QNetwork(state_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=buffer_size)
        self.update_target_every = 100  # steps
        self.learn_step_counter = 0

    def select_action(self, state_hv, valid_actions):
        """
        state_hv: numpy array shape (D,), HDC state
        valid_actions: list of neighbor nodes (เราเลือกแบบ epsilon-greedy)
        """
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # ใช้ state เดียวกันสำหรับทุก action (เพราะ env encode action ผ่าน transition แล้ว)
        # ถ้าอยากให้ action-specific feature ต้องออกแบบเพิ่ม ตอนนี้เอาแบบง่ายก่อน
        state_tensor = torch.from_numpy(state_hv.astype(np.float32)).to(self.device)
        # คำนวณ Q-value เดียว แล้วใช้มันเลือก action แบบ random (เพราะ Q ไม่แยก action)
        # ดังนั้นเพื่อให้มี action-dependence เราจะเพิ่ม noise เล็กน้อยแบบ random
        # -> สำหรับ prototype นี้เรายังใช้ Q เป็นตัวบอก "คุณภาพ state" มากกว่า action-specific

        q_value = self.q_net(state_tensor.unsqueeze(0)).item()
        # เลือก action แบบสุ่ม แต่ bias ด้วย q_value ยังไม่ได้ เพราะโมเดลนี้เป็น state-value
        # ดังนั้นตอนนี้ยังเหมือน random + learning state value (demo concept)
        return random.choice(valid_actions)

    def store_transition(self, state, reward, next_state, done):
        self.replay_buffer.append((state, reward, next_state, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.stack(states).astype(np.float32)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states).astype(np.float32)).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)

        q_values = self.q_net(states)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return loss.item()