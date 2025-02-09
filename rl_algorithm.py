import random

import torch
import torch.nn as nn

from config import config


class RLAlgorithm:
    def __init__(self, environment, agents):
        self.environment = environment
        self.agents = agents
        self.discount_factor = config.DISCOUNT_FACTOR
        self.batch_size = config.BATCH_SIZE

    def train_agent(self, agent, state, action, reward, next_state):
        agent.store_experience(state, action, reward, next_state)

        if len(agent.memory) < self.batch_size:
            return

        batch = random.sample(agent.memory, self.batch_size)
        states = torch.tensor([s[0] for s in batch], dtype=torch.float32, device=agent.device)
        rewards = torch.tensor([[s[2]] for s in batch], dtype=torch.float32, device=agent.device)
        next_states = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=agent.device)

        current_q_values = agent.network(states)
        next_q_values = agent.network(next_states).detach()
        target_q_values = rewards + self.discount_factor * next_q_values

        agent.optimizer.zero_grad()
        loss = nn.MSELoss()(current_q_values, target_q_values)
        loss.backward()
        agent.optimizer.step()

        agent.loss_history.append(loss.item())
        agent.update_epsilon()
