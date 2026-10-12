# 🐍 Snake RL — High-Performance Pure CPU Deep Q-Learning

A stunning, mathematically transparent reinforcement learning showcase running **100% on the CPU** using native JavaScript and `Float32Array` matrix math. An artificial neural network learns to play classic Snake from scratch in real time—with zero GPU dependencies, zero external library overhead, and zero cheats.

🌐 **Live Demo on GitHub Pages**: [https://abhishek-hardaha.github.io/Snake_RL/](https://abhishek-hardaha.github.io/Snake_RL/)

---

## ⚡ Why 100% Pure CPU?

In reinforcement learning with compact state representations (16 inputs, 3 actions), dispatching tiny neural network forward passes and mini-batches to a GPU introduces substantial WebGL/WebGPU overhead (buffer uploads, shader pipelines, asynchronous texture readbacks).

By handcrafting the **Deep Q-Network (DQN)** in native JavaScript with cache-friendly `Float32Array` dot products:
- **Forward Pass Latency**: `< 0.003 ms` (under 3 microseconds!)
- **Mini-Batch Backprop + Adam Step**: `< 0.04 ms`
- **Turbo Simulation Speed**: **1,000+ steps per second** purely on CPU
- **Compatibility**: Runs instantly on any laptop, desktop, or mobile phone without GPU driver issues, shader compilation lag, or WebGL context losses.

---

## 🌟 Key Showcase Features

### 1. 🎮 Single Simulation Centerpiece
A single, focused, beautifully rendered arena window replaces chaotic multi-agent swarms:
- **High-DPI Retina Canvas**: Crisp rendering with subtle ambient floor grid lines.
- **Directional Animated Eyes**: Eyes look in the heading direction and anticipate planned turns.
- **Pulsing Apple**: Pulsing aura with floating sparkle particles.
- **Perception Whiskers**: Optional live raycast beams projecting obstacle clearance ahead, right, and left.
- **Micro-Animations**: Eat sparkles, collision shockwaves, and grid completion progress bars.

### 2. 📈 The Last 30 Runs Performance (Main Screen)
Prominently featured right alongside the main game arena:
- **Live 30-Run Rolling Summary**:
  - 🍎 **Last 30 Avg Score**: Rolling average apples per game.
  - 🏆 **Last 30 Peak Score**: Highest score in the current window.
  - ⏱️ **Avg Survival Steps**: Average steps survived per run.
  - ⚡ **Food Efficiency**: Apples harvested per 100 steps.
  - 📈 **Learning Pace Trend**: Real-time surge / acceleration indicator.
- **Interactive Performance Bar Chart**:
  - 30 dynamic bars color-coded by performance (Gold for master runs, Cyan for solid runs, Violet for exploration).
  - Smooth **10-Run Moving Average Trendline** overlaid across the bars.
  - High-water mark benchmark line.
  - **Interactive Hover Tooltip**: Hover any bar to see exact Episode #, Score, Steps, Length, ε, and Cause of Death.
- **Recent Runs History Feed**:
  - Scrollable log of the last 30 completed episodes with score pills and death classifications (`Wall`, `Self`, `Starvation`).

### 3. 🧠 Live Neural Brain Activity
- Real-time animated canvas visualizing all 4 layers:
  - **Input Layer**: 16 sensory perception neurons.
  - **Hidden Layer 1**: 64 neurons with real live ReLU activation glow.
  - **Hidden Layer 2**: 32 neurons with real live ReLU activation glow.
  - **Output Layer**: 3 Q-values ($Q(\text{Straight})$, $Q(\text{Right})$, $Q(\text{Left})$).

### 4. 🕹️ Comprehensive Interactive Controls
- **▶ / ⏸ Play & Pause**: Toggle live stepping.
- **⚡ Turbo Train**: Uncap CPU simulation throughput to simulate hundreds of steps per second.
- **🏆 Master Brain**: Instant loading of trained champion weights with space-aware lookahead protection.
- **🌀 Grid Sweeper Mode**: Guaranteed 100% win mode (Hamiltonian cycle + safe shortcuts) that fills all 64 grid cells (63/63 score) and continuously trains the DQN via experience replay.
- **🔄 Reset**: Re-randomize weights with He-Normal initialization.
- **Speed Slider**: Smooth variable throttling from 5 to 100 steps/second.
- **🔊 Web Audio Procedural Synth**: Pure procedural chimes for food, crashes, and triumphant victory arpeggio.

---

## 🌀 Grid Sweeping Learning Mode & Breaking the Score 30 Plateau

Standard RL Snake models typically plateau around score 25–30 on an $8\times8$ board (where the snake occupies ~50% of the board) because 1D raycasts cannot foresee non-convex body coils and self-entrapment.

This project solves the plateau via two complementary systems:
1. **🌀 Grid Sweeper Mode (100% Win Guarantee)**:
   - Uses an infallible Hamiltonian cycle covering all 64 tiles without self-intersection.
   - Accelerates early game via tail-distance safe shortcuts, then switches to strict cycle weaving when grid density exceeds 65%.
   - **Learning Mode**: Every $(s, a, r, s', \text{done})$ transition is pushed to the Experience Replay buffer, training the neural network on high-density, winning game states.
2. **Space-Aware Lookahead Trap Suppression & Guided Exploration**:
   - Evaluates connected reachable open space (BFS flood fill) before committing to a greedy Q-action when snake length $> 16$, preventing greedy dives into dead pockets.
   - During exploration ($\epsilon > 0$), mixes 60% Hamiltonian sweeper guidance with safe exploratory moves to prevent suicidal turns and expose the network to long-snake configurations.

---

## 📐 State Vector Specification (16 Egocentric Sensors)

| Index | Feature | Range | Meaning |
|---|---|---|---|
| `[0]` | `danger_straight` | $\{0, 1\}$ | Immediate obstacle 1 step straight ahead |
| `[1]` | `danger_right` | $\{0, 1\}$ | Immediate obstacle 1 step to the relative right |
| `[2]` | `danger_left` | $\{0, 1\}$ | Immediate obstacle 1 step to the relative left |
| `[3..6]` | `heading_one_hot` | $\{0, 1\}$ | Current heading compass vector (`UP`, `RIGHT`, `DOWN`, `LEFT`) |
| `[7..10]` | `food_quadrant` | $\{0, 1\}$ | Relative food bearing (`UP`, `RIGHT`, `DOWN`, `LEFT`) |
| `[11..12]` | `norm_delta_xy` | $[-1, 1]$ | Normalized Manhattan coordinates: $\frac{x_{\text{food}} - x_{\text{head}}}{N}$, $\frac{y_{\text{food}} - y_{\text{head}}}{N}$ |
| `[13..15]` | `raycasts` | $[0, 1]$ | Normalized continuous obstacle clearance (Straight, Right, Left) |

---

## 🚀 Running Locally

Clone the repository and serve with any local HTTP server:

```bash
git clone https://github.com/Abhishek-Hardaha/Snake_RL.git
cd Snake_RL
python -m http.server 8000
```

Open your browser at `http://localhost:8000/`.

---

## 🐍 Standalone Python Script

A PyTorch companion script is included that mirrors the exact same 16-feature architecture:

```bash
python snake_dqn.py
```

---

## 📄 License

MIT License. Created by [Abhishek Hardaha](https://github.com/Abhishek-Hardaha).
