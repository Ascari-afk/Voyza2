"""
src/env_routing.py
==================
Gym-style Routing Environment สำหรับ Bangkok Road Graph

State  : HDC hypervector ขนาด D=10,000 มิติ (float32 numpy array)
         = bundle ของ hypervectors ทุก edge ที่ agent เดินมาแล้ว
           + bind กับ hypervector ของ current node (เพื่อเข้ารหัสตำแหน่ง)

Action : index ใน list ของ neighbor nodes ที่ยังไม่เคยเยือน
         (dynamic per step — ขนาด action space เปลี่ยนได้)

Reward : +100   ถึง destination สำเร็จ
         -edge_length_normalized  ทุก step (ลงโทษตามระยะทาง)
         -5     ถ้าเดินวนซ้ำ node ที่เคยเยือนแล้ว
         -10    หมด max_steps โดยไม่ถึง destination

Interface ที่ run_train_dqn.py ต้องการ:
    env = RoutingEnv(G_real, edge_hv, max_steps=30)
    state = env.reset()                       → np.ndarray shape (D,)
    actions = env.get_valid_actions()         → list[int]  (indices)
    next_state, reward, done, info = env.step(action_idx)
    env.max_steps                             → int
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .hdc_encoding import D, bundle, bind, random_hv


class RoutingEnv:
    """
    Routing Environment บน Bangkok Road Graph

    Parameters
    ----------
    G : nx.Graph
        กราฟถนนจาก build_road_graph_from_roads_gdf()
        แต่ละ edge ต้องมี attribute 'length'
    edge_hv : dict
        mapping (u, v) → hypervector np.ndarray shape (D,)
        จาก build_edge_hv_dict()
    max_steps : int
        จำนวน step สูงสุดต่อ episode
    seed : int | None
        random seed สำหรับ reproducibility
    """

    # ค่า reward
    REWARD_GOAL          = +100.0
    REWARD_REVISIT       = -5.0
    REWARD_TIMEOUT       = -10.0
    STEP_PENALTY_SCALE   = 0.01   # reward = -length * scale ต่อ step

    def __init__(
        self,
        G: nx.Graph,
        edge_hv: Dict[Tuple, np.ndarray],
        max_steps: int = 30,
        seed: Optional[int] = None,
    ):
        self.G        = G
        self.edge_hv  = edge_hv
        self.max_steps = max_steps
        self.rng      = random.Random(seed)
        self.np_rng   = np.random.default_rng(seed)

        # list ของ nodes ทั้งหมด (ตายตัว ใช้ index)
        self._all_nodes: List[Any] = list(G.nodes())
        self._n_nodes = len(self._all_nodes)

        # pre-compute node hypervectors (random, fixed per instance)
        # ใช้ bind กับ state เพื่อบอก "ตอนนี้อยู่ที่ไหน"
        rng_hv = np.random.default_rng(0)
        self._node_hv: Dict[Any, np.ndarray] = {
            n: rng_hv.choice([-1, 1], size=D).astype(np.float32)
            for n in self._all_nodes
        }

        # สถานะ episode (จะ init ใน reset())
        self.current_node: Any    = None
        self.goal_node:    Any    = None
        self.visited:      set    = set()
        self.step_count:   int    = 0
        self._state:       np.ndarray = np.zeros(D, dtype=np.float32)

        # normalize length สำหรับ step penalty
        lengths = [
            d.get("length", 1.0)
            for _, _, d in G.edges(data=True)
        ]
        self._max_length = float(max(lengths)) if lengths else 1.0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        สุ่ม origin + destination ใหม่ (ให้ path มีอยู่จริง)
        คืน initial state (hypervector ของ origin node)
        """
        # สุ่ม origin/goal คู่ที่ connected กัน
        for _ in range(200):
            s, g = self.rng.sample(self._all_nodes, 2)
            if nx.has_path(self.G, s, g):
                self.current_node = s
                self.goal_node    = g
                break
        else:
            # fallback: ใช้ node แรกสองตัว
            self.current_node = self._all_nodes[0]
            self.goal_node    = self._all_nodes[1]

        self.visited    = {self.current_node}
        self.step_count = 0

        # state เริ่มต้น = hypervector ของ origin node
        self._state = self._node_hv[self.current_node].copy()
        return self._state

    # ------------------------------------------------------------------
    # get_valid_actions
    # ------------------------------------------------------------------

    def get_valid_actions(self) -> List[int]:
        """
        คืน list ของ action index (0-based) สำหรับ neighbor ที่เดินได้
        action index → self._neighbors[i]  (เซต per step)
        """
        neighbors = list(self.G.neighbors(self.current_node))
        # เรียงลำดับให้ deterministic (เพื่อให้ agent map index ตรงกัน)
        neighbors.sort(key=str)
        self._neighbors = neighbors   # เก็บไว้ให้ step() ใช้
        return list(range(len(neighbors)))

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Parameters
        ----------
        action_idx : int
            index ใน list ที่ get_valid_actions() คืนมา

        Returns
        -------
        next_state : np.ndarray  shape (D,)
        reward     : float
        done       : bool
        info       : dict
        """
        if not hasattr(self, "_neighbors") or not self._neighbors:
            # ไม่มี action ให้ทำ → timeout
            return self._state.copy(), self.REWARD_TIMEOUT, True, {"reason": "no_actions"}

        # clamp action index (ป้องกัน out-of-range จาก DQN)
        action_idx = int(action_idx) % max(len(self._neighbors), 1)
        next_node  = self._neighbors[action_idx]

        self.step_count += 1

        # ------ คำนวณ reward ------
        reward = 0.0

        # ลงโทษระยะทาง edge ที่เดิน (normalized)
        edge_data = self.G.get_edge_data(self.current_node, next_node, default={})
        edge_length = float(edge_data.get("length", 1.0))
        reward -= edge_length * self.STEP_PENALTY_SCALE

        # ลงโทษ revisit
        if next_node in self.visited:
            reward += self.REWARD_REVISIT

        # ย้ายไป next_node
        prev_node = self.current_node
        self.current_node = next_node
        self.visited.add(next_node)

        # ------ อัปเดต state (HDC) ------
        # state ใหม่ = bundle( state เก่า, bind(edge_hv, node_hv_ปัจจุบัน) )
        edge_key = (prev_node, next_node)
        rev_key  = (next_node, prev_node)
        if edge_key in self.edge_hv:
            hv_edge = self.edge_hv[edge_key]
        elif rev_key in self.edge_hv:
            hv_edge = self.edge_hv[rev_key]
        else:
            hv_edge = self._node_hv.get(next_node, np.zeros(D, dtype=np.float32))

        hv_node     = self._node_hv[next_node]
        hv_step     = bind(hv_edge, hv_node).astype(np.float32)

        # bundle กับ state เก่า → state ใหม่
        new_state = bundle([self._state, hv_step]).astype(np.float32)
        self._state = new_state

        # ------ เช็ค done ------
        done = False
        info: Dict[str, Any] = {
            "current_node": next_node,
            "goal_node":    self.goal_node,
            "step":         self.step_count,
            "visited":      len(self.visited),
        }

        if next_node == self.goal_node:
            reward += self.REWARD_GOAL
            done    = True
            info["reason"] = "reached_goal"
        elif self.step_count >= self.max_steps:
            reward += self.REWARD_TIMEOUT
            done    = True
            info["reason"] = "timeout"

        return self._state.copy(), reward, done, info

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return D

    def __repr__(self) -> str:
        return (
            f"RoutingEnv(nodes={self._n_nodes}, "
            f"max_steps={self.max_steps}, "
            f"current={self.current_node}, goal={self.goal_node})"
        )
