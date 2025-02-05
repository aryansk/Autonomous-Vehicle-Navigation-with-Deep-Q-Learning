# 🚗 Autonomous Vehicle Navigation with Deep Q-Learning

A PyTorch implementation of Deep Q-Network (DQN) for training an autonomous vehicle to navigate through obstacles in a 2D environment, featuring real-time visualization and comprehensive performance metrics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.0-green)
![Pygame](https://img.shields.io/badge/Pygame-2.5.0-red)
![License](https://img.shields.io/badge/License-MIT-purple)

## 🎯 Project Overview

This project implements a Deep Q-Learning agent that learns to navigate a vehicle through a field of obstacles to reach a goal position. The environment features real-time visualization using Pygame and comprehensive training metrics visualization using Matplotlib and Seaborn.

## 🌟 Features

### 🎮 Environment
- Custom Gymnasium environment with Pygame visualization
- Real-time rendering of vehicle, obstacles, and goal
- On-screen metrics display (episode, step, reward, epsilon)
- Configurable screen dimensions and object sizes

### 📊 Visualization & Metrics
- Real-time training visualization
- Comprehensive training metrics plotting:
  - Episode scores with moving average
  - Epsilon decay curve
  - Success rate tracking
- Path heatmap generation
- Interactive display with FPS control

### 🤖 AI Agent
- Deep Q-Network with experience replay
- Epsilon-greedy exploration strategy
- Soft target network updates
- Configurable hyperparameters

## 🛠️ Technical Details

### Environment Parameters
```python
CONSTANTS = {
    'SCREEN_WIDTH': 800,
    'SCREEN_HEIGHT': 600,
    'VEHICLE_SIZE': 20,
    'OBSTACLE_SIZE': 30,
    'N_OBSTACLES': 10
}

COLORS = {
    'BLUE': (0, 0, 255),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLACK': (0, 0, 0),
    'WHITE': (255, 255, 255)
}
```

### Agent Architecture
```python
DQN_ARCHITECTURE = {
    'input_layer': state_size,
    'hidden_layer_1': 128,
    'hidden_layer_2': 64,
    'output_layer': action_size,
    'activation': 'ReLU'
}
```

### Training Parameters
```python
TRAINING_PARAMS = {
    'n_episodes': 1000,
    'buffer_size': 10000,
    'batch_size': 64,
    'gamma': 0.99,
    'tau': 1e-3,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995
}
```

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Required packages:
```
numpy
torch
gymnasium
pygame
matplotlib
seaborn
```

## 🚀 Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autonomous-vehicle-dqn.git
cd autonomous-vehicle-dqn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run training with visualization:
```bash
python main.py
```

## 📊 Visualization Components

### Real-time Display
- Vehicle position and movement
- Obstacle placement
- Goal location
- Current episode metrics
- Training parameters

### Training Metrics
- Episode scores plot
- Moving average score
- Epsilon decay visualization
- Success rate tracking

### Analysis Tools
- Path heatmap generation
- Visitation frequency analysis
- Obstacle influence visualization

## 📁 Project Structure

```
├── main.py
├── requirements.txt
├── models/
│   └── checkpoint.pth
├── visualizations/
│   ├── training_metrics.png
│   └── path_heatmap.png
└── results/
    └── training_stats.csv
```

## 🔍 Classes

### Visualizer
- Handles Pygame initialization and rendering
- Real-time metrics display
- Frame rate control
- Environment state visualization

### VehicleEnv
- Custom Gymnasium environment
- Obstacle generation and management
- Collision detection
- Reward calculation

### DQNAgent
- Neural network management
- Experience replay handling
- Training and inference logic
- Hyperparameter management

## 📈 Performance Metrics

### Training Metrics
- Episode reward
- Moving average score (100 episodes)
- Exploration rate decay
- Success rate tracking

### Visualization Metrics
- Path heatmap
- Obstacle influence
- Goal reaching patterns
- Navigation efficiency

## 🔧 Configuration

### Environment Configuration
```python
env_config = {
    'render_mode': 'human',  # or None for headless training
    'screen_size': (800, 600),
    'fps': 30
}
```

### Training Configuration
```python
training_config = {
    'n_episodes': 1000,
    'plot_interval': 10,
    'save_interval': 100
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Gymnasium documentation
- PyTorch tutorials
- Pygame community
- DQN paper (Mnih et al., 2015)

## 👤 Author

Your Name
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 📞 Support

For support, email your.email@example.com or open an issue in the GitHub repository.

## 📚 References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning
2. Pygame documentation
3. Gymnasium environment guide
4. PyTorch DQN tutorial

---

Made with 🚗 and 🧠 by Aryan Singh
