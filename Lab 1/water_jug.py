"""Water Jug Problem solver using Breadth First Search (state-space search).

State  : (x, y) -> litres of water currently in Jug X and Jug Y.
Goal   : either jug contains exactly `target` litres.
Rules  : fill a jug, empty a jug, pour one jug into the other.
"""

import time
from collections import deque
from math import gcd


def is_solvable(cap_x, cap_y, target):
    """Return True if the target can be measured with the given jugs."""
    if target > max(cap_x, cap_y):
        return False
    return target % gcd(cap_x, cap_y) == 0


def get_next_states(state, cap_x, cap_y):
    """Apply all production rules on `state`; return (new_state, action) pairs."""
    x, y = state
    pour_x_to_y = min(x, cap_y - y)
    pour_y_to_x = min(y, cap_x - x)

    return [
        ((cap_x, y), "Fill Jug X"),
        ((x, cap_y), "Fill Jug Y"),
        ((0, y), "Empty Jug X"),
        ((x, 0), "Empty Jug Y"),
        ((x - pour_x_to_y, y + pour_x_to_y), "Pour Jug X -> Jug Y"),
        ((x + pour_y_to_x, y - pour_y_to_x), "Pour Jug Y -> Jug X"),
    ]


def is_goal(state, target):
    """Goal test: any jug holds exactly `target` litres."""
    return target in state


def solve_water_jug(cap_x, cap_y, target):
    """Find the shortest sequence of actions using BFS.

    Returns (path, nodes_explored). `path` is a list of (state, action)
    from the start state to the goal, or None if no solution exists.
    """
    start = (0, 0)
    queue = deque([start])
    # parent[state] = (previous_state, action_that_led_here)
    parent = {start: (None, "Start")}
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        if is_goal(current, target):
            return build_path(parent, current), nodes_explored

        for next_state, action in get_next_states(current, cap_x, cap_y):
            if next_state not in parent:  # skip already visited states
                parent[next_state] = (current, action)
                queue.append(next_state)

    return None, nodes_explored


def build_path(parent, goal_state):
    """Backtrack through `parent` to rebuild the path from start to goal."""
    path = []
    state = goal_state
    while state is not None:
        previous, action = parent[state]
        path.append((state, action))
        state = previous
    path.reverse()
    return path


def print_solution(path):
    """Display the solution step by step."""
    print(f"\n{'Step':<6}{'Action':<24}{'(Jug X, Jug Y)'}")
    print("-" * 46)
    for step, (state, action) in enumerate(path):
        print(f"{step:<6}{action:<24}{state}")


def read_int(prompt, minimum):
    """Read an integer >= minimum from the user, retrying on invalid input."""
    while True:
        try:
            value = int(input(prompt))
            if value < minimum:
                print(f"Please enter a number >= {minimum}.")
                continue
            return value
        except ValueError:
            print("Invalid input! Please enter a whole number.")


def main():
    print("=== Water Jug Problem (BFS) ===")
    cap_x = read_int("Capacity of Jug X (litres): ", 1)
    cap_y = read_int("Capacity of Jug Y (litres): ", 1)
    target = read_int("Target amount (litres): ", 0)

    if not is_solvable(cap_x, cap_y, target):
        print("\nNo solution possible for these values.")
        print("(Target must be <= larger jug and divisible by GCD of capacities.)")
        return

    start_time = time.perf_counter()
    path, nodes_explored = solve_water_jug(cap_x, cap_y, target)
    elapsed = time.perf_counter() - start_time

    if path is None:
        print("\nNo solution found.")
        return

    print_solution(path)
    print(f"\nSteps in shortest solution : {len(path) - 1}")
    print(f"States explored            : {nodes_explored}")
    print(f"Execution time             : {elapsed:.6f} seconds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")