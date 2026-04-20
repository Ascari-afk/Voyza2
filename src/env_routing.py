import random
import numpy as np
import networkx as nx

from .hdc_encoding import build_edge_hv_dict, bundle, D

class RoutingEnv:
    """
    RL environment อย่างง่าย:
      - กราฟ G (undirected)
      - มี start_node, goal_node
      - state = hypervector ที่ bundle จาก edge HV รอบ node ปัจจุบัน (หรือจาก last edge)
      - action = เลือกเพื่อนบ้าน node ถัดไป
      - episode จบเมื่อถึง goal หรือเกิน max_steps
    """

    def __init__(self, G: nx.Graph, edge_hv: dict, max_steps: int = 50):
        self.G = G
        # edge_hv คีย์อาจเป็น (u, v) หรือ (v, u) ต้อง normalize
        self.edge_hv = {}
        for (u, v), hv in edge_hv.items():
            self.edge_hv[(u, v)] = hv
            self.edge_hv[(v, u)] = hv  # ทำให้ใช้ได้สองทิศ

        self.max_steps = max_steps

        self.start = None
        self.goal = None
        self.current = None
        self.steps = 0

    def sample_start_goal(self):
        nodes = list(self.G.nodes())
        self.start, self.goal = random.sample(nodes, 2)

    def reset(self):
        self.sample_start_goal()
        self.current = self.start
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        """
        state = hypervector ที่แทน context ของ node ปัจจุบัน:
          - bundle ของ edge HV ทั้งหมดที่ออกจาก node นี้
        """
        hvs = []
        for neighbor in self.G.neighbors(self.current):
            hv = self.edge_hv.get((self.current, neighbor), None)
            if hv is not None:
                hvs.append(hv)

        if len(hvs) == 0:
            # ถ้าไม่มีเพื่อนบ้าน (ไม่น่าเกิดบ่อย) ใช้เวกเตอร์ศูนย์
            return np.zeros(D, dtype=int)

        state_hv = bundle(hvs)
        return state_hv

    def get_valid_actions(self):
        """
        คืน list ของเพื่อนบ้านที่เป็น action ได้ (node id)
        """
        return list(self.G.neighbors(self.current))

    def step(self, action_node):
        """
        action_node = node ปลายที่จะไปจาก current (ต้องเป็นเพื่อนบ้าน)
        คืนค่า (next_state, reward, done, info)
        """
        if action_node not in self.G[self.current]:
            # action ไม่ถูกต้อง → ลงโทษหนัก
            reward = -10.0
            done = True
            info = {"invalid_action": True}
            return self._get_state(), reward, done, info

        data = self.G[self.current][action_node]
        length = float(data.get("length", 1.0))
        # หา key num_buildings_...m
        bkeys = [k for k in data.keys() if str(k).startswith("num_buildings_") and str(k).endswith("m")]
        if len(bkeys) > 0:
            num_b = float(data.get(bkeys[0], 0.0))
        else:
            num_b = 0.0

        # ตัวอย่าง reward function: ลงโทษระยะยาว + ตึกเยอะ (สมมติว่าตึกเยอะ = โอกาสรถติดมาก)
        alpha = 0.1
        reward = - length - alpha * num_b

        self.current = action_node
        self.steps += 1

        done = False
        info = {}

        if self.current == self.goal:
            reward += 100.0  # bonus ถึงเป้าหมาย
            done = True
            info["reason"] = "goal_reached"
        elif self.steps >= self.max_steps:
            done = True
            info["reason"] = "max_steps"

        next_state = self._get_state()
        return next_state, reward, done, info