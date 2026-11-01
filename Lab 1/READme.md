# Lab 1: Water Jug Problem

 1. Aim / Problem Statement ::

Given two jugs of capacities X and Y litres (no measuring marks), measure exactly
**T litres** of water using only these operations: fill a jug, empty a jug, or pour
water from one jug into the other. The problem is solved using state-space search.

 2. Algorithm ::

**State representation:** `(x, y)` = litres of water in Jug X and Jug Y. Start state is `(0, 0)`.

**Production rules:**

| Rule | Action |
|---|---|
| 1 | Fill Jug X |
| 2 | Fill Jug Y |
| 3 | Empty Jug X |
| 4 | Empty Jug Y |
| 5 | Pour Jug X into Jug Y (until Y is full or X is empty) |
| 6 | Pour Jug Y into Jug X (until X is full or Y is empty) |

**Goal test:** either jug contains exactly T litres.

**Steps (Breadth First Search):**
1. Check solvability: T <= max(X, Y) and T is divisible by GCD(X, Y).
2. Put start state `(0, 0)` in a queue.
3. Remove a state from the queue; if it is the goal, stop.
4. Apply all 6 rules to generate next states; add unvisited states to the queue and store their parent.
5. Repeat until goal is found, then backtrack through parents to print the path.

BFS explores level by level, so the solution found has the **minimum number of steps**.

3. Results / Output ::

Input: Jug X = 4, Jug Y = 3, Target = 2

| Step | Action | (Jug X, Jug Y) |
|---|---|---|
| 0 | Start | (0, 0) |
| 1 | Fill Jug Y | (0, 3) |
| 2 | Pour Jug Y -> Jug X | (3, 0) |
| 3 | Fill Jug Y | (3, 3) |
| 4 | Pour Jug Y -> Jug X | (4, 2) |


