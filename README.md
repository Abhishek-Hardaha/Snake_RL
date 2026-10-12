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
- **▶ / ⏸ Play & Pause** (Keyboard shortcut: `Space`).
- **⏭ Step Once**: Advance exactly 1 frame for step-by-step decision analysis.
- **🎯 Pure Greedy Mode ($\epsilon=0$)**: Test the learned policy without random exploratory mistakes.
- **⚡ Turbo Train**: Uncap CPU simulation throughput to train hundreds of episodes in seconds.
- **🏆 Load Master Brain**: Instant 1-click loading of pre-trained champion weights (scores 18–31+ apples).
- **🔄 Train From Scratch**: Randomize weights with He-Normal initialization to watch learning emerge live.
- **Speed Slider**: Smooth variable throttling from 1 to 120 steps/second.
- **Grid Size Selector**: $6\times6$, $8\times8$ (default), or $10\times10$.
- **Theme Switcher**: `Cyber Neon`, `Matrix Emerald`, `Synthwave Sunset`, `Obsidian Gold`.
- **🔊 Web Audio Procedural Synth**: Harmonic eat chimes and crash sounds with zero audio file downloads.

---

## 🛡️ The Zero-Cheating Guarantee

- **Pure Neural Policy**: Every action is either an exact forward pass $\arg\max_a Q(s, a)$ or explicit $\epsilon$-greedy exploration.
- **Zero Heuristics**: No A* search, no BFS, no Hamiltonian cycles, no artificial steering.
- **Strict Classic Rules**: Instant death on wall hit or self-collision; exactly 1 food spawned uniformly at random on vacant cells.

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
