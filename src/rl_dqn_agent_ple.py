"""
src/rl_dqn_agent.py
===================
DQN Agent สำหรับ RoutingEnv

Architecture
------------
State dim : D = 10,000  (HDC hypervector, bipolar float32)
Action    : dynamic per step (ขนาดไม่คงที่)
            → Q-network คำนวณ Q-value ต่อ action ทีละ action
              แล้วเลือก argmax จาก valid_actions เท่านั้น

Network   : Input(D) → Linear(512) → LayerNorm → ReLU
                     → Linear(256) → LayerNorm → ReLU
                     → Linear(1)   (Q-value ของ action นั้น)
            * รับ state concat action_idx_onehot ขนาด MAX_ACTION_DIM

เนื่องจาก action space ของแต่ละ step ไม่คงที่ (neighbor count ต่างกัน)
เราใช้แนวทาง "action-in" DQN:
    input = [state_hv (D,) || action_onehot (MAX_ACTION_DIM,)]
    output = scalar Q-value
แล้ว evaluate ทุก valid action → เลือก argmax

ถ้า MAX_ACTION_DIM = 8 ก็รองรับ node degree ≤ 8
(config.py ตั้ง MAX_DEGREE = 6 ดังนั้น 8 เผื่อไว้)

Interface ที่ run_train_dqn.py ต้องการ:
    agent = DQNAgent(state_dim=D)
    action = agent.select_action(state, valid_actions)   → int
    agent.store_transition(state, reward, next_state, done)
    loss   = agent.train_step()                          → float | None
    agent.epsilon                                        → float
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------- Hyperparameters ----------

MAX_ACTION_DIM  = 8       # รองรับ degree สูงสุด (MAX_DEGREE=6, เผื่อ 2)
HIDDEN_1        = 512
HIDDEN_2        = 256
LEARNING_RATE   = 1e-3
GAMMA           = 0.95    # discount factor
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
EPSILON_DECAY   = 0.995   # คูณทุก episode (call decay_epsilon() ทุก episode)
REPLAY_CAPACITY = 10_000
BATCH_SIZE      = 64
TARGET_UPDATE_FREQ = 200  # อัปเดต target network ทุก N steps


# ---------- Q-Network ----------

class QNetwork(nn.Module):
    """
    Action-in DQN: input = [state || action_onehot] → scalar Q
    ใช้ LayerNorm แทน BatchNorm เพื่อรองรับ batch size เล็ก/ไม่คงที่
    """

    def __init__(self, state_dim: int, action_dim: int = MAX_ACTION_DIM):
        super().__init__()
        in_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_1),
            nn.LayerNorm(HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.LayerNorm(HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, state_dim + action_dim) → (batch, 1)"""
        return self.net(x)


# ---------- Replay Buffer ----------

class ReplayBuffer:
    """
    Circular replay buffer เก็บ transition (state, action_idx, reward, next_state, done)
    state / next_state เป็น numpy float32 shape (D,)
    """

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action_idx: int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.append((
            state.astype(np.float32),
            int(action_idx),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),           # (B, D)
            np.array(actions),          # (B,)
            np.array(rewards, dtype=np.float32),   # (B,)
            np.stack(next_states),      # (B, D)
            np.array(dones, dtype=np.float32),     # (B,)
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------- DQN Agent ----------

class DQNAgent:
    """
    DQN Agent ที่ใช้ action-in Q-network

    Parameters
    ----------
    state_dim : int
        ขนาด state vector (= D = 10,000)
    action_dim : int
        ขนาด action one-hot (= MAX_ACTION_DIM = 8)
    device : str | None
        'cuda', 'mps', หรือ 'cpu'  (auto-detect ถ้า None)
    """

    def __init__(
        self,
        state_dim:  int = 10_000,
        action_dim: int = MAX_ACTION_DIM,
        device:     Optional[str] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device     = torch.device(device)
        self.state_dim  = state_dim
        self.action_dim = action_dim

        print(f"  [DQNAgent] device={self.device}  state_dim={state_dim}  action_dim={action_dim}")

        # online network + target network (Double DQN)
        self.q_net      = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.replay    = ReplayBuffer(REPLAY_CAPACITY)

        self.epsilon   = EPSILON_START
        self._step_count = 0   # global step (สำหรับ target update)

    # ------------------------------------------------------------------
    # One-hot helper
    # ------------------------------------------------------------------

    def _action_onehot(self, action_idx: int) -> np.ndarray:
        """คืน float32 one-hot ขนาด action_dim"""
        oh = np.zeros(self.action_dim, dtype=np.float32)
        oh[action_idx % self.action_dim] = 1.0
        return oh

    def _make_input(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        """concat [state(D), action_onehot(action_dim)] → (D+action_dim,)"""
        return np.concatenate([state.astype(np.float32), self._action_onehot(action_idx)])

    # ------------------------------------------------------------------
    # select_action (ε-greedy)
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        Parameters
        ----------
        state         : np.ndarray (D,)  current state
        valid_actions : list[int]         action indices จาก env.get_valid_actions()

        Returns
        -------
        int  action index ที่เลือก
        """
        if not valid_actions:
            return 0

        # ε-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Greedy: evaluate Q ทุก valid action แล้วเลือก max
        self.q_net.eval()
        with torch.no_grad():
            q_values = []
            for a in valid_actions:
                inp = self._make_input(state, a)
                x   = torch.tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
                q   = self.q_net(x).item()
                q_values.append(q)

        best_idx = int(np.argmax(q_values))
        return valid_actions[best_idx]

    # ------------------------------------------------------------------
    # store_transition
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state:      np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        action_idx: int = 0,   # run_train_dqn.py ไม่ส่ง action แยก → default 0
    ) -> None:
        """
        เก็บ transition ลง replay buffer

        หมายเหตุ: run_train_dqn.py เรียกว่า
            agent.store_transition(state, reward, next_state, done)
        → action ไม่ถูกส่งมา → ใช้ action=0 เป็น default
        """
        self.replay.push(state, action_idx, reward, next_state, done)

    # ------------------------------------------------------------------
    # train_step (sample + update)
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """
        Sample batch จาก replay buffer แล้วทำ gradient step

        Returns
        -------
        float loss หรือ None ถ้า buffer ยังไม่พอ
        """
        if len(self.replay) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)

        # แปลงเป็น tensor
        states_t      = torch.tensor(states,      dtype=torch.float32, device=self.device)  # (B,D)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards_t     = torch.tensor(rewards,     dtype=torch.float32, device=self.device)  # (B,)
        dones_t       = torch.tensor(dones,       dtype=torch.float32, device=self.device)
        actions_t     = torch.tensor(actions,     dtype=torch.long,    device=self.device)  # (B,)

        # สร้าง one-hot tensor สำหรับ action ที่ stored
        action_oh = torch.zeros(BATCH_SIZE, self.action_dim, dtype=torch.float32, device=self.device)
        idx_clamped = actions_t.clamp(0, self.action_dim - 1)
        action_oh.scatter_(1, idx_clamped.unsqueeze(1), 1.0)

        # Q(s, a) จาก online network
        inp_online = torch.cat([states_t, action_oh], dim=1)   # (B, D+action_dim)
        self.q_net.train()
        q_pred = self.q_net(inp_online).squeeze(1)             # (B,)

        # Target Q: Double DQN
        # 1) online net เลือก best action ใน next_state
        # 2) target net ประเมิน Q ของ action นั้น
        with torch.no_grad():
            # evaluate ทุก action dim ใน next state ด้วย online net → เลือก argmax
            all_q_online = []
            for a in range(self.action_dim):
                oh = torch.zeros(BATCH_SIZE, self.action_dim, dtype=torch.float32, device=self.device)
                oh[:, a] = 1.0
                inp_a = torch.cat([next_states_t, oh], dim=1)
                q_a   = self.q_net(inp_a).squeeze(1)
                all_q_online.append(q_a)

            all_q_online = torch.stack(all_q_online, dim=1)   # (B, action_dim)
            best_actions = all_q_online.argmax(dim=1)          # (B,)

            # target net ประเมิน
            best_oh = torch.zeros(BATCH_SIZE, self.action_dim, dtype=torch.float32, device=self.device)
            best_oh.scatter_(1, best_actions.unsqueeze(1), 1.0)
            inp_target = torch.cat([next_states_t, best_oh], dim=1)
            q_target_next = self.target_net(inp_target).squeeze(1)  # (B,)

            q_target = rewards_t + GAMMA * q_target_next * (1.0 - dones_t)

        # Huber loss (robust ต่อ outlier reward)
        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping ป้องกัน exploding gradient กับ D=10000
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # อัปเดต target network แบบ hard update
        self._step_count += 1
        if self._step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    # epsilon decay (เรียกจาก training loop ทุก episode)
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """ลด epsilon แบบ exponential decay"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "step_count": self._step_count,
        }, path)
        print(f"  [DQNAgent] saved to {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt.get("epsilon",    EPSILON_MIN)
        self._step_count = ckpt.get("step_count", 0)
        print(f"  [DQNAgent] loaded from {path}  epsilon={self.epsilon:.3f}")

    def __repr__(self) -> str:
        return (
            f"DQNAgent(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"epsilon={self.epsilon:.3f}, "
            f"replay={len(self.replay)}/{self.replay.buffer.maxlen}, "
            f"device={self.device})"
        )
