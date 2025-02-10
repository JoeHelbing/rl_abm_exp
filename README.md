# Reinforcement Learning Enhanced Schelling Segregation Model

This project implements an enhanced version of Schelling's segregation model using reinforcement learning (RL) agents. The simulation combines classical agent-based modeling with modern deep learning techniques to study emergent social patterns. This implementation has each individual agent instatiate and learn on a separate nueral network.

## Background

### Schelling Segregation Model
Thomas Schelling's model (1971) demonstrates how individual preferences can lead to systematic segregation, even when individuals are relatively tolerant. In the traditional model, agents prefer to live in neighborhoods where some fraction of their neighbors are of the same type. The model shows how these micro-level preferences can create macro-level patterns of segregation.

### RL Enhancement
This implementation enhances the classical model by:
- Replacing simple rule-based decisions with learned behaviors through Deep Q-Learning
- Allowing agents to learn optimal strategies for maximizing their happiness
- Introducing a continuous reward system based on neighborhood composition
- Implementing a torus topology for the grid to eliminate edge effects

## Components

### Core Modules

- `environment.py`: Implements the grid environment with torus topology
- `agent.py`: Defines the RL agent with Deep Q-Network architecture
- `rl_algorithm.py`: Contains the reinforcement learning training logic
- `simulation.py`: Manages the simulation lifecycle and metrics collection
- `config.py`: Centralizes configuration parameters

### Web Interface

- `web_server.py`: Flask server for the web interface
- `templates/`: HTML templates for visualization
- `static/`: CSS styles and JavaScript visualization code

### Key Features

- Interactive web visualization of the simulation
- Real-time metrics tracking
- Configurable parameters through web interface
- Agent-specific metrics visualization
- Grid visualization with agent types

## Technical Details

### Agent Architecture
Each agent uses a neural network with the following structure:
- Input: Current position, agent type, and grid state
- Hidden layers: Configurable through `config.py`
- Output: Q-values for possible actions

### Learning Process
- Agents learn through experience using Q-learning
- Epsilon-greedy exploration strategy
- Experience replay for stable learning
- Continuous reward based on neighborhood happiness

### Metrics Tracked
- Average happiness per agent type
- Training loss
- Exploration rate (epsilon) decay
- Individual agent metrics

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the web server:
   ```bash
   python web_server.py
   ```

3. Open a web browser and navigate to `http://localhost:5000`

## Configuration

Key parameters can be adjusted through the web interface or by modifying `config.py`:
- Grid size
- Number of agents per type
- Learning parameters (learning rate, epsilon decay, etc.)
- Neural network architecture
- Reward function parameters

## Requirements

- Python 3.10+
- PyTorch
- Flask
- Flask-SocketIO
- NumPy

## Visualization

The web interface provides:
- Real-time grid visualization
- Happiness trends by agent type
- Learning metrics (loss, epsilon decay)
- Individual agent performance tracking