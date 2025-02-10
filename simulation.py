import random

import numpy as np

from environment import Environment
from rl_algorithm import RLAlgorithm


class Simulation:
    def __init__(self, grid_size=10, num_agents_per_type=10):
        self.grid_size = grid_size
        self.num_agents_per_type = num_agents_per_type
        self.environment = Environment(grid_size, num_agents_per_type)
        self.agents = self.environment.agents
        self.rl_algorithm = RLAlgorithm(self.environment, self.agents)
        self.current_episode = 0
        self.metrics_history = []
        self.previous_grid = None  # Store the previous grid state

    def run_episode(self):
        self.current_episode += 1

        # Create empty dictionaries to record initial states and chosen actions
        initial_states = {}
        actions = {}
        new_positions = {}

        # Randomly shuffle the agent list so they choose one-by-one in random order
        agent_list = list(self.agents)
        random.shuffle(agent_list)

        # Each agent chooses an action in turn and updates the board immediately.
        for agent in agent_list:
            # Initialize agent's previous visible area if it's None
            if agent.previous_grid is None:
                agent.previous_grid = self.environment.get_visible_area(agent.position)

            # Get current state features for the agent's current position
            current_state = agent.get_state_features(agent.position)
            initial_states[agent] = current_state

            # Get the valid empty positions based on adjacent positions
            valid_positions = self.environment.get_empty_adjacent_positions(
                agent.position
            )
            if valid_positions:
                chosen_position = agent.choose_action(valid_positions)
                actions[agent] = chosen_position
                new_positions[agent] = chosen_position  # Save the move explicitly
                self.environment.move_agent(agent, chosen_position)
            else:
                actions[agent] = agent.position
                new_positions[agent] = agent.position

            # Update agent's previous grid state with current visible area
            agent.previous_grid = self.environment.get_visible_area(agent.position)

            # Train the agent immediately after its move
            reward = agent.calculate_reward(new_positions[agent])
            next_state = agent.get_state_features(new_positions[agent])
            self.rl_algorithm.train_agent(
                agent, initial_states[agent], new_positions[agent], reward, next_state
            )

        metrics = self.calculate_metrics()
        self.metrics_history.append(metrics)
        return metrics

    def calculate_metrics(self):
        type1_agents = [a for a in self.agents if a.agent_type == 1]
        type2_agents = [a for a in self.agents if a.agent_type == 2]

        metrics = {
            "average_happiness_type1": float(
                np.mean([a.calculate_happiness(a.position) for a in type1_agents])
            ),
            "average_happiness_type2": float(
                np.mean([a.calculate_happiness(a.position) for a in type2_agents])
            ),
            "average_epsilon_type1": float(np.mean([a.epsilon for a in type1_agents])),
            "average_epsilon_type2": float(np.mean([a.epsilon for a in type2_agents])),
            "average_lr": float(np.mean([a.current_lr for a in self.agents])),
        }

        # Calculate agent-specific metrics
        agent_metrics = []
        for idx, agent in enumerate(self.agents):
            agent_metrics.append(
                {
                    "id": idx,
                    "type": agent.agent_type,
                    "happiness": float(agent.calculate_happiness(agent.position)),
                    "loss": float(
                        agent.current_loss
                    ),  # Use current_loss instead of loss_history
                    "learning_rate": float(agent.current_lr),
                }
            )
            print(
                f"Agent {idx} metrics - Type: {agent.agent_type}, Loss: {agent.current_loss}"
            )  # Debug print

        metrics["agent_metrics"] = agent_metrics
        return metrics

    def get_current_state(self):
        return {
            "grid": self.environment.grid,
            "episode": self.current_episode,
            "metrics": self.metrics_history[-1] if self.metrics_history else None,
        }
