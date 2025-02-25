"""Configuration parameters for the simulation"""

import json
from pathlib import Path
from typing import Any, Dict


class Config:
    def __init__(self):
        # Agent parameters
        self.INITIAL_LEARNING_RATE = 0.001  # Starting learning rate
        self.MIN_LEARNING_RATE = 0.0001  # Final learning rate
        self.WARMUP_STEPS = 10  # Reduced warmup steps
        self.TOTAL_STEPS = 100  # Learning rate reaches minimum at this step
        self.INITIAL_EPSILON = 1
        self.EPSILON_MIN = 0.1
        self.EPSILON_TARGET_STEP = 100  # Step at which epsilon reaches minimum
        self.EPSILON_DECAY = (
            self.INITIAL_EPSILON - self.EPSILON_MIN
        ) / self.EPSILON_TARGET_STEP

        # Neural network parameters - Updated for exact 5x5 visible area dimensions
        # Input: 3 features (x,y,type) + 25 cells (current view) + 25 cells (previous view) = 53
        self.NETWORK_LAYERS = [
            (53, 64),  # Input layer to first hidden layer
            (64, 64),  # First to second hidden layer
            (64, 32),  # Second to third hidden layer
            (32, 1),  # Third hidden layer to output
        ]

        # RL parameters
        self.DISCOUNT_FACTOR = 0.99
        self.BATCH_SIZE = 8  # Reduced batch size to start learning earlier
        self.MEMORY_SIZE = 1000

        # Reward parameters
        self.HAPPINESS_THRESHOLD = 0.5  # Threshold for agent happiness

    def load_from_file(self, config_path: str) -> None:
        """Load configuration from a JSON file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            config_dict = json.load(f)
            self._update_from_dict(config_dict)

    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to a JSON file"""
        config_dict = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and key.isupper()
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key) and key.isupper():
                setattr(self, key, value)


# Create global configuration instance
config = Config()
