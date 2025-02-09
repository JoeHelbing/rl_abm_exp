import numpy as np

from agent import Agent


class Environment:
    def __init__(self, grid_size, num_agents_per_type):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.num_agents_per_type = num_agents_per_type
        self.agents = self.initialize_agents()

    def get_all_empty_positions(self):
        """Get all empty positions on the grid"""
        empty_y, empty_x = np.where(self.grid == 0)
        return list(zip(empty_y, empty_x))

    def _place_agent(self, agent_type):
        """Find empty position and place agent"""
        empty_positions = np.where(self.grid == 0)
        if len(empty_positions[0]) == 0:
            raise ValueError("No empty positions available on grid")

        idx = np.random.randint(len(empty_positions[0]))
        x, y = empty_positions[0][idx], empty_positions[1][idx]

        self.grid[x, y] = agent_type
        return (x, y)

    def initialize_agents(self):
        """Initialize agents of both types on the grid"""
        agents = []
        for agent_type in [1, 2]:
            agents.extend(
                [
                    Agent(self._place_agent(agent_type), agent_type, self)
                    for _ in range(self.num_agents_per_type)
                ]
            )
        return agents

    def move_agent(self, agent, new_position):
        """Move agent to new position and update grid"""
        x, y = agent.position
        new_x, new_y = new_position
        self.grid[x, y] = 0
        self.grid[new_x, new_y] = agent.agent_type
        agent.position = (new_x, new_y)
        return (new_x, new_y)

    def get_neighbors(self, position):
        """Get neighboring positions with torus topology (wrapping around edges)"""
        x, y = position
        neighbors = []
        # Check all 8 neighboring positions
        for dx, dy in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            # Apply torus wrapping by using modulo arithmetic
            nx = (x + dx) % self.grid_size
            ny = (y + dy) % self.grid_size
            neighbors.append((nx, ny))
        return neighbors

    def get_empty_neighbors(self, position):
        """Get neighboring positions that are empty"""
        neighbors = self.get_neighbors(position)
        return [n for n in neighbors if self.grid[n] == 0]

    def get_neighbor_types(self, positions):
        """Get the agent types at the given positions"""
        return [self.grid[x, y] for x, y in positions]

    def get_grid_state(self):
        """Return current grid state"""
        return self.grid.copy()
