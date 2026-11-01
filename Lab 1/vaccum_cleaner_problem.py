"""Vacuum Cleaner Problem: agent simulation in a two-room world.

Environment : two rooms, A and B, each Clean or Dirty.
Percept     : (location, status) of the room the agent is in.
Actions     : Left, Right, Suck, NoOp.
Performance : +10 for cleaning a dirty room, -1 for each move,
              -1 for sucking in an already clean room, 0 for NoOp.

Three agents are compared:
    1. Random agent        - picks any action at random
    2. Simple reflex agent - decides only from the current percept
    3. Model-based agent   - remembers rooms it has already cleaned
"""

import random

ACTIONS = ["Left", "Right", "Suck"]


class VacuumEnvironment:
    """Two-room world that applies agent actions and tracks the score."""

    def __init__(self, location, status_a, status_b):
        self.location = location
        self.rooms = {"A": status_a, "B": status_b}
        self.score = 0

    def percept(self):
        """What the agent can sense: its location and that room's status."""
        return self.location, self.rooms[self.location]

    def execute(self, action):
        """Apply `action`, update the score and return the reward."""
        if action == "Suck":
            if self.rooms[self.location] == "Dirty":
                self.rooms[self.location] = "Clean"
                reward = 10
            else:
                reward = -1
        elif action == "Right":
            self.location = "B"
            reward = -1
        elif action == "Left":
            self.location = "A"
            reward = -1
        elif action == "NoOp":
            reward = 0
        else:
            raise ValueError(f"Unknown action: {action}")
        self.score += reward
        return reward


class RandomAgent:
    """Chooses an action randomly, ignoring the percept."""

    name = "Random Agent"

    def act(self, percept):
        return random.choice(ACTIONS)


class SimpleReflexAgent:
    """Condition-action rules based only on the current percept."""

    name = "Simple Reflex Agent"

    def act(self, percept):
        location, status = percept
        if status == "Dirty":
            return "Suck"
        return "Right" if location == "A" else "Left"


class ModelBasedAgent:
    """Keeps an internal model (memory) of the rooms it has visited."""

    name = "Model-Based Agent"

    def __init__(self):
        self.memory = {"A": None, "B": None}

    def act(self, percept):
        location, status = percept
        self.memory[location] = status
        if status == "Dirty":
            self.memory[location] = "Clean"  # room will be clean after Suck
            return "Suck"
        if all(state == "Clean" for state in self.memory.values()):
            return "NoOp"  # both rooms known clean -> stop wasting moves
        return "Right" if location == "A" else "Left"


def run_simulation(agent, env, steps, verbose=True):
    """Run `agent` in `env` for `steps` steps; return the final score."""
    if verbose:
        print(f"\n--- {agent.name} ---")
        print(f"{'Step':<6}{'Percept':<20}{'Action':<8}{'Reward':<8}{'Score'}")
    for step in range(1, steps + 1):
        percept = env.percept()
        action = agent.act(percept)
        reward = env.execute(action)
        if verbose:
            print(f"{step:<6}{str(percept):<20}{action:<8}{reward:<8}{env.score}")
    return env.score


def compare_agents(agent_classes, steps, trials):
    """Average score of each agent over `trials` random starting worlds."""
    totals = {cls.name: 0 for cls in agent_classes}
    for _ in range(trials):
        location = random.choice(["A", "B"])
        status_a = random.choice(["Clean", "Dirty"])
        status_b = random.choice(["Clean", "Dirty"])
        for cls in agent_classes:
            env = VacuumEnvironment(location, status_a, status_b)
            totals[cls.name] += run_simulation(cls(), env, steps, verbose=False)
    return {name: total / trials for name, total in totals.items()}


def read_choice(prompt, valid_options):
    """Keep asking until the user enters one of `valid_options`."""
    while True:
        value = input(prompt).strip().upper()
        if value in valid_options:
            return value
        print(f"Invalid input! Choose from: {', '.join(valid_options)}")


def main():
    print("=== Vacuum Cleaner Problem ===")
    location = read_choice("Starting room of vacuum (A/B): ", ["A", "B"])
    status_a = "Dirty" if read_choice("Is room A dirty? (Y/N): ", ["Y", "N"]) == "Y" else "Clean"
    status_b = "Dirty" if read_choice("Is room B dirty? (Y/N): ", ["Y", "N"]) == "Y" else "Clean"
    steps = 8

    random.seed(42)  # reproducible results for the report
    agent_classes = [RandomAgent, SimpleReflexAgent, ModelBasedAgent]

    for cls in agent_classes:
        env = VacuumEnvironment(location, status_a, status_b)
        run_simulation(cls(), env, steps)

    trials = 1000
    print(f"\n=== Performance comparison ({trials} random worlds, {steps} steps each) ===")
    averages = compare_agents(agent_classes, steps, trials)
    for name, average in averages.items():
        print(f"{name:<24}: average score = {average:.2f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")