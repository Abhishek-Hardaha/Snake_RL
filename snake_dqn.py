"""
=============================================================================
SNAKE DEEP Q-LEARNING (DQN) — STANDALONE PYTHON / PYTORCH IMPLEMENTATION
=============================================================================
This script mirrors the exact 16-feature architecture, state observation vector, 
reward structure, and Bellman update used in the high-performance CPU web application.
Zero cheating: no hardcoded pathfinding, no heuristics, pure RL.

Requirements:
    pip install torch numpy

Usage:
    python snake_dqn.py
=============================================================================
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Direction vectors: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
DIRS = [
    (0, -1), # UP
    (1, 0),  # RIGHT
    (0, 1),  # DOWN
    (-1, 0)  # LEFT
]

class SnakeGame:
    def __init__(self, grid_size=8, seed=42):
        self.grid_size = grid_size
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.dir = 1 # Start moving RIGHT
        mid = self.grid_size // 2
        self.body = [(mid, mid)]
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.is_dead = False
        self.has_won = False
        self.spawn_food()
        return self.get_state()

    def spawn_food(self):
        empty = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.body
        ]
        if not empty:
            self.has_won = True
            self.food = None
            return
        self.food = self.rng.choice(empty)

    def step(self, relative_action):
        # 0: Straight, 1: Turn Right, 2: Turn Left
        if relative_action == 1:
            self.dir = (self.dir + 1) % 4
        elif relative_action == 2:
            self.dir = (self.dir + 3) % 4

        self.steps += 1
        self.steps_without_food += 1

        dx, dy = DIRS[self.dir]
        hx, hy = self.body[0]
        new_head = (hx + dx, hy + dy)

        # Collision checks
        hit_wall = not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size)
        hit_body = new_head in self.body[1:]

        if hit_wall or hit_body:
            self.is_dead = True
            return -10.0, True, self.get_state()

        self.body.insert(0, new_head)

        # Food check
        if self.food and new_head == self.food:
            self.score += 1
            self.steps_without_food = 0
            if len(self.body) == self.grid_size * self.grid_size:
                self.has_won = True
                return 100.0, True, self.get_state()
            self.spawn_food()
            return 10.0, False, self.get_state()

        self.body.pop()

        # Starvation loop prevention
        if self.steps_without_food > max(100, len(self.body) * 60):
            self.is_dead = True
            return -5.0, True, self.get_state()

        # Dense reward shaping
        old_dist = abs(hx - self.food[0]) + abs(hy - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        reward = 0.15 if new_dist < old_dist else -0.20

        return reward, False, self.get_state()

    def get_state(self):
        hx, hy = self.body[0]
        straight = DIRS[self.dir]
        right = DIRS[(self.dir + 1) % 4]
        left = DIRS[(self.dir + 3) % 4]

        def is_danger(x, y):
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                return 1.0
            if (x, y) in self.body[:-1]:
                return 1.0
            return 0.0

        def raycast(dx, dy):
            cx, cy = hx + dx, hy + dy
            dist = 0
            while 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                if (cx, cy) in self.body[:-1]:
                    break
                dist += 1
                cx += dx
                cy += dy
            return dist / self.grid_size

        fx, fy = (self.food if self.food else (hx, hy))

        state = [
            # 0..2: Danger sensors
            is_danger(hx + straight[0], hy + straight[1]),
            is_danger(hx + right[0],    hy + right[1]),
            is_danger(hx + left[0],     hy + left[1]),
            # 3..6: Heading one-hot
            1.0 if self.dir == 0 else 0.0,
            1.0 if self.dir == 1 else 0.0,
            1.0 if self.dir == 2 else 0.0,
            1.0 if self.dir == 3 else 0.0,
            # 7..10: Food quadrant
            1.0 if fy < hy else 0.0,
            1.0 if fx > hx else 0.0,
            1.0 if fy > hy else 0.0,
            1.0 if fx < hx else 0.0,
            # 11..12: Normalized distance delta
            (fx - hx) / self.grid_size,
            (fy - hy) / self.grid_size,
            # 13..15: Continuous free space raycasts
            raycast(straight[0], straight[1]),
            raycast(right[0], right[1]),
            raycast(left[0], left[1]),
        ]
        return np.array(state, dtype=np.float32)


class QNetwork(nn.Module):
    def __init__(self, input_dim=16, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, gamma=0.95, lr=0.001, batch_size=64):
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.994

        self.device = torch.device("cpu") # Optimized for high-throughput CPU
        self.model = QNetwork().to(self.device)
        self.target_model = QNetwork().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss() # Huber Loss
        self.memory = deque(maxlen=20000)
        self.updates = 0

    def act(self, state, force_greedy=False):
        eps = 0 if force_greedy else self.epsilon
        if random.random() < eps:
            return random.randint(0, 2)
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s_t = torch.as_tensor(np.array(states, dtype=np.float32), device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.as_tensor(np.array(next_states, dtype=np.float32), device=self.device)
        d_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.model(s_t).gather(1, a_t)

        with torch.no_grad():
            max_next_q = self.target_model(ns_t).max(1)[0].unsqueeze(1)
            target_q = r_t + (1 - d_t) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        if self.updates % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train():
    print("=" * 65)
    print("SNAKE DQN CPU TRAINING — 100% GENUINE REINFORCEMENT LEARNING")
    print("=" * 65)

    env = SnakeGame(grid_size=8, seed=42)
    agent = DQNAgent()

    episodes = 500
    scores = []
    best_score = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            reward, done, next_state = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            agent.train_step()
            state = next_state

        agent.decay_epsilon()
        scores.append(env.score)
        if env.score > best_score:
            best_score = env.score

        if ep % 20 == 0 or env.has_won:
            avg_score = np.mean(scores[-20:])
            print(f"Episode {ep:4d} | Score: {env.score:2d} | Avg(20): {avg_score:4.1f} | Best: {best_score:2d} | ε: {agent.epsilon:.3f} | Buffer: {len(agent.memory):5d}")

        if env.has_won:
            print(f"\n🎉 FULL VICTORY! Board completely filled at Episode {ep}!")
            break

if __name__ == "__main__":
    train()
