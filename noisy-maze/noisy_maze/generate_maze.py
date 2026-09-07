"""Maze construction copied from maze/generate_maze.py; observation noise is applied afterwards."""

import random
from collections import deque
from decimal import ROUND_FLOOR, Decimal

import numpy as np


class MazeGenerator:
    def __init__(self, size=7, seed=None, algorithm="prim"):
        self.algorithm = algorithm
        if algorithm not in ["prim", "dfs"]:
            raise ValueError("algorithm must be prim or dfs")
        if size < 5 or size % 2 != 1:
            raise ValueError("size must be an odd integer >= 5")
        self.size = size
        self.rng = random.Random(seed)
        self.grid = np.ones((size, size), dtype=int)
        self.start = (1, 1)
        self.goal = (size - 2, size - 2)
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    def generate(self):
        self.grid.fill(1)
        if self.algorithm == "dfs":
            self.carve_passages_from(self.start[0], self.start[1])
        elif self.algorithm == "prim":
            self.prim()
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        return self.grid

    def prim(self):
        start_x, start_y = self.start[0], self.start[1]
        self.grid[start_x, start_y] = 0
        frontier = []
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        for dx, dy in directions:
            nx, ny = start_x + dx, start_y + dy
            if 0 < nx < self.size - 1 and 0 < ny < self.size - 1:
                if self.grid[nx, ny] == 1:
                    frontier.append((nx, ny))
        while frontier:
            idx = self.rng.randint(0, len(frontier) - 1)
            fx, fy = frontier.pop(idx)
            neighbors = []
            for dx, dy in directions:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx, ny] == 0:
                        neighbors.append((nx, ny))
            if neighbors:
                nx, ny = self.rng.choice(neighbors)
                mid_x, mid_y = (fx + nx) // 2, (fy + ny) // 2
                self.grid[mid_x, mid_y] = 0
                self.grid[fx, fy] = 0
                for dx, dy in directions:
                    new_x, new_y = fx + dx, fy + dy
                    if 0 < new_x < self.size - 1 and 0 < new_y < self.size - 1:
                        if self.grid[new_x, new_y] == 1 and (new_x, new_y) not in frontier:
                            frontier.append((new_x, new_y))

    def carve_passages_from(self, cx, cy):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.rng.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            mx, my = cx + 2 * dx, cy + 2 * dy
            if 0 <= mx < self.size and 0 <= my < self.size:
                if self.grid[mx, my] == 1:
                    self.grid[nx, ny] = 0
                    self.grid[mx, my] = 0
                    self.carve_passages_from(mx, my)

    def solve_bfs(self):
        queue = deque([(self.start, [])])
        visited = set([self.start])
        while queue:
            (cx, cy), path = queue.popleft()
            if (cx, cy) == self.goal:
                return path
            for action_idx, (dx, dy) in self.actions.items():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx, ny] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        new_path = path + [action_idx]
                        queue.append(((nx, ny), new_path))
        return None

    def to_text_sequence(self):
        optimal_actions = self.solve_bfs()
        if optimal_actions is None:
            raise ValueError("Generated maze has no solution")
        grid_tokens = []
        for r in range(self.size):
            for c in range(self.size):
                pos = (r, c)
                if pos == self.start:
                    grid_tokens.append("START")
                elif pos == self.goal:
                    grid_tokens.append("GOAL")
                elif self.grid[r, c] == 1:
                    grid_tokens.append("WALL")
                else:
                    grid_tokens.append("PATH")
            grid_tokens.append("NEWLINE")
        action_tokens = [self.action_names[a] for a in optimal_actions]
        sequence = ["<bos>", "GRID_START"] + grid_tokens + ["GRID_END", "PATH_START"] + action_tokens + ["DONE", "<eos>"]
        text_sequence = " ".join(sequence)
        return {"sequence": text_sequence, "optimal_path_length": len(optimal_actions)}


def normalize_noise_fraction(value: str) -> str:
    fraction = Decimal(str(value))
    if not fraction.is_finite() or not 0 <= fraction <= 1:
        raise ValueError("noise_fraction must be between 0 and 1")
    return format(fraction.normalize(), "f")


def connecting_positions(size: int) -> list[tuple[int, int]]:
    """Interior edges between odd/odd cell centers, excluding the border."""
    if size < 5 or size % 2 != 1:
        raise ValueError("size must be an odd integer >= 5")
    return [(r, c) for r in range(1, size - 1) for c in range(1, size - 1) if r % 2 != c % 2]


def obscure_observation(item: dict, size: int, noise_fraction: str, rng: random.Random) -> dict:
    """Mask an exact number of edges without changing the real grid or solution.

    The topology generator and mask sampler use independent random streams.
    `sequence` is the model input plus the true path; `ground_truth` is never masked.
    """
    fraction = normalize_noise_fraction(noise_fraction)
    positions = connecting_positions(size)
    count = int((Decimal(fraction) * len(positions)).to_integral_value(rounding=ROUND_FLOOR))
    masked_positions = sorted(rng.sample(positions, count))
    tokens = item["sequence"].split()
    grid_start = tokens.index("GRID_START") + 1
    for r, c in masked_positions:
        index = grid_start + r * (size + 1) + c
        if tokens[index] not in {"WALL", "PATH"}:
            raise ValueError(f"Expected WALL or PATH at connecting position {(r, c)}")
        tokens[index] = "UNKNOWN"
    return {
        "sequence": " ".join(tokens),
        "ground_truth": item["sequence"],
        "optimal_path_length": item["optimal_path_length"],
        "noise_fraction": float(fraction),
        "masked_positions": masked_positions,
    }
