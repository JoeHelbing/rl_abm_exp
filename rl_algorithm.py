import random

import torch
import torch.nn as nn

from config import config

torch.set_float32_matmul_precision("high")


class RLAlgorithm:
    def __init__(self, environment, agents):
        self.environment = environment
        self.agents = agents
        self.discount_factor = config.DISCOUNT_FACTOR
        self.batch_size = config.BATCH_SIZE

    def train_agent(self, agent, state, action, reward, next_state):
        agent.store_experience(state, action, reward, next_state)

        # Start learning as soon as we have enough samples for a minimal batch
        current_batch_size = min(len(agent.memory), self.batch_size)
        if current_batch_size < 1:  # Allow single-sample learning initially
            agent.current_loss = 0.0
            return

        # Even with one sample, we can still learn from it
        batch = random.sample(agent.memory, current_batch_size)
        states = torch.tensor(
            [s[0] for s in batch], dtype=torch.float32, device=agent.device
        )
        rewards = torch.tensor(
            [[s[2]] for s in batch], dtype=torch.float32, device=agent.device
        )
        next_states = torch.tensor(
            [s[3] for s in batch], dtype=torch.float32, device=agent.device
        )

        # Validate tensor dimensions
        expected_dim = config.NETWORK_LAYERS[0][0]
        if states.shape[-1] != expected_dim:
            raise ValueError(
                f"States dimension mismatch. Expected {expected_dim}, got {states.shape[-1]}"
            )
        if next_states.shape[-1] != expected_dim:
            raise ValueError(
                f"Next states dimension mismatch. Expected {expected_dim}, got {next_states.shape[-1]}"
            )

        try:
            current_q_values = agent.network(states)
            next_q_values = agent.network(next_states).detach()
            target_q_values = rewards + self.discount_factor * next_q_values

            agent.optimizer.zero_grad()
            loss = nn.MSELoss()(current_q_values, target_q_values)
            loss.backward()
            agent.optimizer.step()

            # Store loss value and update epsilon/learning rate
            agent.current_loss = loss.item()  # Store current loss in agent
            agent.loss_history.append(agent.current_loss)
            agent.update_epsilon()
            agent.update_learning_rate()  # Make sure learning rate is updated after each training step

        except Exception as e:
            print(f"Error during training: {str(e)}")
            print(
                f"States shape: {states.shape}, Next states shape: {next_states.shape}"
            )
            raise
