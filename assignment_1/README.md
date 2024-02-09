# Assignment 1

A grid-world simulation for Assignment 1.

## Command Line Usage

- mode `enum` -> Mode of the simulation.
  - manual -> Manually runs a single configurable simulation.
    - map_seed `u64` -> Seed of the random map generation. Using the same seed will result in the same map.
    - size x `i32` -> Width of the map.
    - size y `i32` -> Height of the map.
    - interval `f32` -> Seconds interval between simulation steps. Lower the value the faster the simulation runs. Intervals <= 0 will run as fast as possible.
    - map_mode `enum` -> Type of map to generate.
      - `dfs` -> Generates a map using DFS. There is only one connected component.
      - `dfs-rand` -> Generates a map using DFS, then random breaks and creates walls. Creates multiple connected components.
    - goal_mode `enum` -> Type of goal to set.
      - `reachable` -> Sets a reachable goal for the agent.
      - `unreachable` -> Sets an unreachable goal for the agent. Only works if the map has more than one connected component.
    - break_tie_mode `enum` -> How to break ties during A* search.
      - `none` -> No tie breaking
      - `high-g` -> Prefers state with higher actual cost (g) 
      - `low-g` -> Prefers state with lower actual cost (g)
    - is_forward `bool` -> Whether A* searches forwards (starting from agent position) or backwards (starting from goal position).
      - Only relevant for `ai_behavior = astar`
    - ai_behavior `enum` -> Pathfinding behavior for the AI
      - `astar` -> Repeated A* algorithm
      - `adaptive-astar` -> Adaptive A* algorithm
  - auto -> Automatically runs some simulation tests and displays aggregate data.
    - map_seed `u64` -> Map seed for the auto tests. Using the same seed will result in the same map.

## Examples

```bash
cargo run manual 1234 51 51 0.05 dfs reachable none forward astar
```
Run a manual test with seed `1234` on a map of size `51x51`. Step the simulation every `0.05` seconds. Run DFS to generate a maze map, and set a reachable goal. Don't use any tie-breaking strategies for A*, and use repeated forward A*.