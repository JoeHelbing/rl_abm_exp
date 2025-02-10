import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from config import config

# Set matmul precision to high for better performance
torch.set_float32_matmul_precision("high")


class AgentNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for input_size, output_size in config.NETWORK_LAYERS[:-1]:
            layers.extend([nn.Linear(input_size, output_size), nn.GELU()])
        input_size, output_size = config.NETWORK_LAYERS[-1]
        layers.append(nn.Linear(input_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        expected_dim = config.NETWORK_LAYERS[0][0]
        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected input dimension {expected_dim}, got {x.shape[-1]}"
            )
        return self.net(x)


class Agent:
    def __init__(self, position, agent_type, environment):
        self.position = position
        self.agent_type = agent_type
        self.environment = environment
        self.loss_history = []
        self.network = AgentNetwork()
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.network.to(self.device)
        # self.network = torch.compile(self.network)
        self.epsilon = config.INITIAL_EPSILON
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.steps = 0  # Track individual agent steps
        self.optimizer = None  # Will be initialized with get_scheduled_lr
        self.initialize_optimizer()
        self.previous_grid = None  # Store the previous visible area state
        self.current_lr = config.INITIAL_LEARNING_RATE
        self.current_loss = 0.0  # Track current loss value

    def initialize_optimizer(self):
        """Initialize optimizer with initial learning rate"""
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=config.INITIAL_LEARNING_RATE
        )

    def get_scheduled_lr(self):
        """Calculate learning rate using linear decay after warmup"""
        if self.steps < config.WARMUP_STEPS:
            return config.INITIAL_LEARNING_RATE
        elif self.steps >= config.TOTAL_STEPS:
            return config.MIN_LEARNING_RATE
        else:
            # Linear decay from INITIAL_LR to MIN_LR
            decay_steps = config.TOTAL_STEPS - config.WARMUP_STEPS
            decay_progress = (self.steps - config.WARMUP_STEPS) / decay_steps
            return (
                config.INITIAL_LEARNING_RATE
                - (config.INITIAL_LEARNING_RATE - config.MIN_LEARNING_RATE)
                * decay_progress
            )

    def update_learning_rate(self):
        """Update the learning rate according to the schedule"""
        self.current_lr = self.get_scheduled_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

    def update_epsilon(self):
        """Update epsilon using linear decay"""
        self.epsilon = max(
            config.EPSILON_MIN,
            config.INITIAL_EPSILON - (config.EPSILON_DECAY * self.steps),
        )
        self.steps += 1

    def calculate_reward(self, new_position):
        """Calculate reward based on improvement in happiness proportion"""
        old_happiness = self.calculate_happiness(self.position)
        new_happiness = self.calculate_happiness(new_position)

        # Base reward is the improvement in happiness
        reward = new_happiness - old_happiness
        return reward

    def calculate_happiness(self, position):
        """Calculate happiness based on Schelling's segregation model"""
        neighbors = self.environment.get_neighbors(position)
        if not neighbors:
            return 0

        neighbor_types = self.environment.get_neighbor_types(neighbors)
        same_type = sum(1 for n in neighbor_types if n == self.agent_type)
        total_neighbors = len(
            [n for n in neighbor_types if n != 0]
        )  # Exclude empty cells

        if total_neighbors == 0:
            return 0

        # Happiness is the proportion of same-type neighbors
        return same_type / total_neighbors

    def store_experience(self, state, action, reward, next_state):
        """Store experience in replay memory and return current metrics"""
        self.memory.append((state, action, reward, next_state))
        return {
            "loss": self.current_loss,
            "lr": self.current_lr,
            "epsilon": self.epsilon,
        }

    def get_state_features(self, position):
        x, y = position
        visible_area = self.environment.get_visible_area(position)
        visible_area_flat = visible_area.flatten()
        previous_visible_area_flat = (
            self.previous_grid.flatten()
            if self.previous_grid is not None
            else visible_area_flat
        )

        # Normalize coordinates to [0,1] range to make learning easier
        norm_x = x / (self.environment.grid_size - 1)
        norm_y = y / (self.environment.grid_size - 1)

        # Validate feature dimensions
        features = (
            [norm_x, norm_y, self.agent_type]
            + list(visible_area_flat)
            + list(previous_visible_area_flat)
        )
        expected_dim = config.NETWORK_LAYERS[0][0]
        if len(features) != expected_dim:
            raise ValueError(
                f"State features dimension mismatch. Expected {expected_dim}, got {len(features)}"
            )
        return features

    def choose_action(self, valid_positions):
        if random.random() < self.epsilon:
            return random.choice(valid_positions)

        states = []
        for pos in valid_positions:
            features = self.get_state_features(pos)
            states.append(features)

        states_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            q_values = self.network(states_tensor)
        print(q_values)
        return valid_positions[torch.argmax(q_values).item()]

    def predict_happiness_increase(self, position):
        """Predict the increase in happiness for a given position"""
        current_happiness = self.calculate_happiness(self.position)
        predicted_happiness = self.calculate_happiness(position)
        return predicted_happiness - current_happiness
