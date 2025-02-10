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

    def get_visible_area(self, position):
        """Get the 5x5 area centered on the agent's position with torus topology"""
        x, y = position
        visible_area = np.zeros((5, 5))
        for i in range(-2, 3):
            for j in range(-2, 3):
                # Apply torus wrapping
                wrapped_x = (x + i) % self.grid_size
                wrapped_y = (y + j) % self.grid_size
                visible_area[i + 2, j + 2] = self.grid[wrapped_x, wrapped_y]
        return visible_area

    def get_adjacent_positions(self, position):
        """Get positions adjacent to the current position (including staying put)"""
        x, y = position
        adjacent = [(x, y)]  # Include current position (stay put)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Apply torus wrapping
            new_x = (x + dx) % self.grid_size
            new_y = (y + dy) % self.grid_size
            adjacent.append((new_x, new_y))
        return adjacent

    def get_empty_adjacent_positions(self, position):
        """Get empty positions adjacent to the current position"""
        adjacent = self.get_adjacent_positions(position)
        return [pos for pos in adjacent if self.grid[pos] == 0 or pos == position]

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

        # Verify the move is valid (adjacent or same position)
        valid_positions = self.get_adjacent_positions((x, y))
        if (new_x, new_y) not in valid_positions:
            return agent.position  # Return current position if move is invalid

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
