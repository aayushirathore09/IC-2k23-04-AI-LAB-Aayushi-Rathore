"""State Space Representation using the two-room Vacuum Cleaner problem.

A state space problem is defined by:
    1. Initial state  : where the agent starts
    2. Actions        : what the agent can do in a given state
    3. Transition     : result of applying an action to a state
    4. Goal test      : is the given state a goal state?

State format: (location, room_A_dirty, room_B_dirty)
    location        -> 'A' or 'B'
    room_X_dirty    -> True (dirty) / False (clean)
"""

from collections import deque


class StateSpaceProblem:
    """Abstract blueprint: any problem is described by these 4 methods."""

    def initial_state(self):
        raise NotImplementedError

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def is_goal(self, state):
        raise NotImplementedError


class VacuumWorld(StateSpaceProblem):
    """Two rooms (A, B); the vacuum can move left/right or suck dirt."""

    def __init__(self, start_state):
        self.start_state = start_state

    def initial_state(self):
        return self.start_state

    def actions(self, state):
        """Return only the actions that are applicable in `state`."""
        location, dirty_a, dirty_b = state
        applicable = []
        if location == "A":
            applicable.append("Right")
        else:
            applicable.append("Left")
        current_room_dirty = dirty_a if location == "A" else dirty_b
        if current_room_dirty:
            applicable.append("Suck")
        return applicable

    def result(self, state, action):
        """Transition function: (state, action) -> new state."""
        location, dirty_a, dirty_b = state
        if action == "Right":
            return ("B", dirty_a, dirty_b)
        if action == "Left":
            return ("A", dirty_a, dirty_b)
        if action == "Suck":
            if location == "A":
                return (location, False, dirty_b)
            return (location, dirty_a, False)
        raise ValueError(f"Unknown action: {action}")

    def is_goal(self, state):
        """Goal: both rooms are clean."""
        _, dirty_a, dirty_b = state
        return not dirty_a and not dirty_b


def format_state(state):
    """Convert a state tuple into a human-readable string."""
    location, dirty_a, dirty_b = state
    room_a = "Dirty" if dirty_a else "Clean"
    room_b = "Dirty" if dirty_b else "Clean"
    return f"[Agent in {location} | A: {room_a} | B: {room_b}]"


def generate_state_space(problem):
    """Explore every reachable state; return dict {state: [(action, next)]}."""
    start = problem.initial_state()
    state_space = {}
    queue = deque([start])

    while queue:
        state = queue.popleft()
        if state in state_space:
            continue
        state_space[state] = []
        for action in problem.actions(state):
            next_state = problem.result(state, action)
            state_space[state].append((action, next_state))
            if next_state not in state_space:
                queue.append(next_state)
    return state_space


def print_state_space(problem, state_space):
    """Print every state with its outgoing transitions."""
    print(f"\nTotal reachable states: {len(state_space)}\n")
    for state, transitions in state_space.items():
        tag = "  <-- GOAL" if problem.is_goal(state) else ""
        print(f"{format_state(state)}{tag}")
        for action, next_state in transitions:
            print(f"    --{action}--> {format_state(next_state)}")


def find_solution(problem):
    """BFS over the state space; return list of (action, state) to a goal."""
    start = problem.initial_state()
    queue = deque([start])
    parent = {start: (None, None)}

    while queue:
        state = queue.popleft()
        if problem.is_goal(state):
            path = []
            while state is not None:
                prev, action = parent[state]
                path.append((action, state))
                state = prev
            return path[::-1]
        for action in problem.actions(state):
            next_state = problem.result(state, action)
            if next_state not in parent:
                parent[next_state] = (state, action)
                queue.append(next_state)
    return None


def read_choice(prompt, valid_options):
    """Keep asking until the user enters one of `valid_options`."""
    while True:
        value = input(prompt).strip().upper()
        if value in valid_options:
            return value
        print(f"Invalid input! Choose from: {', '.join(valid_options)}")


def main():
    print("=== State Space Representation: Vacuum Cleaner World ===")
    location = read_choice("Starting room of vacuum (A/B): ", ["A", "B"])
    dirty_a = read_choice("Is room A dirty? (Y/N): ", ["Y", "N"]) == "Y"
    dirty_b = read_choice("Is room B dirty? (Y/N): ", ["Y", "N"]) == "Y"

    problem = VacuumWorld((location, dirty_a, dirty_b))

    print(f"\nInitial state: {format_state(problem.initial_state())}")
    state_space = generate_state_space(problem)
    print_state_space(problem, state_space)

    solution = find_solution(problem)
    print("\n--- Solution path (BFS) ---")
    for step, (action, state) in enumerate(solution):
        label = action if action else "Start"
        print(f"Step {step}: {label:<6} -> {format_state(state)}")
    print(f"\nTotal actions needed: {len(solution) - 1}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")