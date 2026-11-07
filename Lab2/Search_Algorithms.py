"""Maze Solver using BFS, DFS, Hill Climbing, Best First Search and A*.

Maze symbols:  S = start, G = goal, # = wall, . = open cell
State          : (row, col) of the agent
Actions        : Up, Down, Left, Right (if the next cell is not a wall)
Heuristic h(n) : Manhattan distance to the goal
"""

import heapq
import os
import time
from collections import deque

MAZE = [
    "S..#......",
    "##.#.####.",
    "........#.",
    ".######.#.",
    ".#....#...",
    ".####.####",
    ".........G",
]

# Row/column change for Up, Right, Down, Left
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# ----------------------------------------------------------------------
# Maze helpers
# ----------------------------------------------------------------------
def validate_maze(maze):
    """Raise ValueError if the maze is empty, uneven or has no S / G."""
    if not maze:
        raise ValueError("Maze is empty.")
    if len({len(row) for row in maze}) != 1:
        raise ValueError("All maze rows must have the same length.")
    symbols = "".join(maze)
    if symbols.count("S") != 1 or symbols.count("G") != 1:
        raise ValueError("Maze must contain exactly one 'S' and one 'G'.")


def find_symbol(maze, symbol):
    """Return the (row, col) position of `symbol` in the maze."""
    for row_index, row in enumerate(maze):
        if symbol in row:
            return row_index, row.index(symbol)
    raise ValueError(f"Symbol '{symbol}' not found in maze.")


def get_neighbors(maze, cell):
    """Return all walkable neighbouring cells of `cell`."""
    row, col = cell
    neighbors = []
    for d_row, d_col in MOVES:
        new_row, new_col = row + d_row, col + d_col
        inside = 0 <= new_row < len(maze) and 0 <= new_col < len(maze[0])
        if inside and maze[new_row][new_col] != "#":
            neighbors.append((new_row, new_col))
    return neighbors


def manhattan(cell, goal):
    """Heuristic: Manhattan distance between two cells."""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def reconstruct_path(parent, goal):
    """Backtrack from goal to start using the `parent` dictionary."""
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    return path[::-1]


# ----------------------------------------------------------------------
# Search algorithms: each returns (path or None, nodes_explored)
# ----------------------------------------------------------------------
def bfs(maze, start, goal):
    """Breadth First Search: explores level by level (shortest path)."""
    queue = deque([start])
    parent = {start: None}
    explored = 0
    while queue:
        cell = queue.popleft()
        explored += 1
        if cell == goal:
            return reconstruct_path(parent, goal), explored
        for neighbor in get_neighbors(maze, cell):
            if neighbor not in parent:
                parent[neighbor] = cell
                queue.append(neighbor)
    return None, explored


def dfs(maze, start, goal):
    """Depth First Search: goes as deep as possible before backtracking."""
    stack = [start]
    parent = {start: None}
    visited = set()
    explored = 0
    while stack:
        cell = stack.pop()
        if cell in visited:
            continue
        visited.add(cell)
        explored += 1
        if cell == goal:
            return reconstruct_path(parent, goal), explored
        for neighbor in get_neighbors(maze, cell):
            if neighbor not in visited:
                parent[neighbor] = cell
                stack.append(neighbor)
    return None, explored


def hill_climbing(maze, start, goal):
    """Hill Climbing: always moves to the neighbour with the lowest h(n).

    It stops (and fails) when no neighbour is better than the current cell,
    i.e. when it gets stuck at a local optimum.
    """
    current = start
    path = [start]
    visited = {start}
    explored = 1
    while current != goal:
        candidates = [n for n in get_neighbors(maze, current) if n not in visited]
        if not candidates:
            return None, explored
        best = min(candidates, key=lambda cell: manhattan(cell, goal))
        if manhattan(best, goal) >= manhattan(current, goal):
            return None, explored  # stuck at a local optimum
        current = best
        visited.add(current)
        path.append(current)
        explored += 1
    return path, explored


def best_first_search(maze, start, goal):
    """Greedy Best First Search: expands the cell with the lowest h(n)."""
    counter = 0  # tie-breaker so the heap never compares cells
    heap = [(manhattan(start, goal), counter, start)]
    parent = {start: None}
    visited = set()
    explored = 0
    while heap:
        _, _, cell = heapq.heappop(heap)
        if cell in visited:
            continue
        visited.add(cell)
        explored += 1
        if cell == goal:
            return reconstruct_path(parent, goal), explored
        for neighbor in get_neighbors(maze, cell):
            if neighbor not in parent:
                parent[neighbor] = cell
                counter += 1
                heapq.heappush(heap, (manhattan(neighbor, goal), counter, neighbor))
    return None, explored


def a_star(maze, start, goal):
    """A* Search: expands the cell with the lowest f(n) = g(n) + h(n)."""
    counter = 0
    heap = [(manhattan(start, goal), counter, start)]
    parent = {start: None}
    cost_so_far = {start: 0}
    visited = set()
    explored = 0
    while heap:
        _, _, cell = heapq.heappop(heap)
        if cell in visited:
            continue
        visited.add(cell)
        explored += 1
        if cell == goal:
            return reconstruct_path(parent, goal), explored
        for neighbor in get_neighbors(maze, cell):
            new_cost = cost_so_far[cell] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                parent[neighbor] = cell
                counter += 1
                priority = new_cost + manhattan(neighbor, goal)
                heapq.heappush(heap, (priority, counter, neighbor))
    return None, explored


ALGORITHMS = {
    "BFS": bfs,
    "DFS": dfs,
    "Hill Climbing": hill_climbing,
    "Best First": best_first_search,
    "A*": a_star,
}


# ----------------------------------------------------------------------
# Display and benchmarking
# ----------------------------------------------------------------------
def print_maze(maze, path=None):
    """Print the maze; cells on `path` are marked with '*'."""
    on_path = set(path) if path else set()
    for row_index, row in enumerate(maze):
        line = ""
        for col_index, symbol in enumerate(row):
            if symbol == "." and (row_index, col_index) in on_path:
                line += "*"
            else:
                line += symbol
        print("  " + " ".join(line))


def benchmark(algorithm, maze, start, goal, repeats=200):
    """Run `algorithm` many times; return (path, explored, avg time in ms)."""
    begin = time.perf_counter()
    for _ in range(repeats):
        path, explored = algorithm(maze, start, goal)
    average_ms = (time.perf_counter() - begin) / repeats * 1000
    return path, explored, average_ms


def print_results_table(results):
    """Print the comparison table of all algorithms."""
    print(f"\n{'Algorithm':<15}{'Path Found':<12}{'Path Length':<13}"
          f"{'Nodes Explored':<16}{'Time (ms)'}")
    print("-" * 66)
    for name, (path, explored, time_ms) in results.items():
        found = "Yes" if path else "No"
        length = len(path) - 1 if path else "-"
        print(f"{name:<15}{found:<12}{str(length):<13}{explored:<16}{time_ms:.4f}")


def save_comparison_chart(results, folder="outputs"):
    """Save a bar chart of nodes explored and time (needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: skipping chart. "
              "Install it with: pip install matplotlib)")
        return

    os.makedirs(folder, exist_ok=True)
    names = list(results)
    explored = [results[name][1] for name in names]
    times = [results[name][2] for name in names]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.bar(names, explored, color="steelblue")
    left.set_title("Nodes Explored")
    right.bar(names, times, color="darkorange")
    right.set_title("Execution Time (ms)")
    for axis in (left, right):
        axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()

    file_path = os.path.join(folder, "comparison.png")
    figure.savefig(file_path, dpi=150)
    print(f"\nChart saved to {file_path}")


def main():
    validate_maze(MAZE)
    start = find_symbol(MAZE, "S")
    goal = find_symbol(MAZE, "G")

    print("=== Maze Solver: Search Algorithm Comparison ===\n")
    print("Maze:")
    print_maze(MAZE)

    results = {}
    for name, algorithm in ALGORITHMS.items():
        path, explored, time_ms = benchmark(algorithm, MAZE, start, goal)
        results[name] = (path, explored, time_ms)
        print(f"\n--- {name} ---")
        if path:
            print(f"Path length: {len(path) - 1}, nodes explored: {explored}")
            print_maze(MAZE, path)
        else:
            print(f"No path found (nodes explored: {explored}).")
            if name == "Hill Climbing":
                print("Reason: stuck at a local optimum (no better neighbour).")

    print_results_table(results)
    save_comparison_chart(results)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"Maze error: {error}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")