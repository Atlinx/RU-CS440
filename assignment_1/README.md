# Assignment 1

A grid-world simulation for Assignment 1.

## Command Line Usage

- mode `enum` -> Mode of the simulation.
  - manual -> Manually runs a single configurable simulation.
    - size x `i32` -> Width of the map.
    - size y `i32` -> Height of the map.
    - interval `f32` -> Seconds interval between simulation steps. Lower the value the faster the simulation runs. Intervals <= 0 will run as fast as possible.
    - map_mode `enum` -> Type of map to generate.
      - `dfs` -> Generates a map using DFS. There is only one connected component.
      - `dfs-rand` -> Generates a map using DFS, then random breaks and creates walls. Creates multiple connected components.
    - goal_mode `enum` -> Type of goal to set.
      - `reachable` -> Sets a reachable goal for the agent.
      - `unreachable` -> Sets an unreachable goal for the agent. Only works if the map has more than one connected component.
    - map_seed `u64` -> Seed of the random map generation. Using the same seed will result in the same map.
  - auto -> Automatically runs some simulation tests and displays aggregate data.
    - map_seed `u64` -> Map seed for the auto tests. Using the same seed will result in the same map.