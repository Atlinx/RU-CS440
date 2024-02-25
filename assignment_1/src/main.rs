#![allow(dead_code)]
use rand::{thread_rng, Rng};
use separator::Separatable;
use std::{env, time::Instant};

use prelude::*;

pub mod agent;
pub mod heap;
pub mod map;
pub mod simulation;
pub mod utils;

pub mod prelude {
    pub use crate::agent::*;
    pub use crate::heap::*;
    pub use crate::map::*;
    pub use crate::simulation::*;
    pub use crate::utils::*;
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut args: Vec<String> = env::args().collect();

    // Remove program name
    args.reverse();
    args.pop().unwrap();
    pub enum Mode {
        HeapTests,
        AutoTests,
        ManualTest,
        AdaptiveExample,
    }
    let mut mode: Mode = Mode::ManualTest;
    if let Some(mode_str) = args.pop() {
        if mode_str.to_lowercase() == "adaptive" {
            mode = Mode::AdaptiveExample;
        } else if mode_str.to_lowercase() == "heap" {
            mode = Mode::HeapTests;
        } else if mode_str.to_lowercase() == "auto" {
            mode = Mode::AutoTests;
        } else if mode_str.to_lowercase() == "manual" {
            mode = Mode::ManualTest;
        }
    }

    match mode {
        Mode::AdaptiveExample => test_adaptive_example(args),
        Mode::HeapTests => heap::heap_tests(args),
        Mode::AutoTests => auto_tests(args),
        Mode::ManualTest => manual_test(args),
    }
}

// region Testing
fn test_adaptive_example(mut args: Vec<String>) {
    let simulation_text = "
_____
_____
__#__
__#__
__A#G
";
    let mut is_adaptive = false;
    if let Some(map_type_str) = args.pop() {
        is_adaptive = map_type_str == "adaptive";
    }

    if is_adaptive {
        if let Some(simulation) = SimulationBuilder::from_text(
            simulation_text,
            Box::new(AdaptiveAStarBehavior::new(BreakTieMode::HigherGCost, true)),
        ) {
            let mut runner = SimulationRunner::new(simulation, 0.5, true);
            runner.run();
        } else {
            println!(
                "⛔ Could not load simulation from text:\n{}",
                simulation_text
            );
        }
    } else {
        if let Some(simulation) = SimulationBuilder::from_text(
            simulation_text,
            Box::new(AStarBehavior::new(true, BreakTieMode::HigherGCost)),
        ) {
            let mut runner = SimulationRunner::new(simulation, 0.5, true);
            runner.run();
        } else {
            println!(
                "⛔ Could not load simulation from text:\n{}",
                simulation_text
            );
        }
    }
}

fn auto_tests(mut args: Vec<String>) {
    let mut rng_seed = thread_rng().gen();
    if let Some(seed_str) = args.pop() {
        if let Ok(seed_val) = seed_str.parse() {
            rng_seed = seed_val;
        }
    }

    let mut trials = 50;
    if let Some(trials_str) = args.pop() {
        if let Ok(trials_val) = trials_str.parse() {
            trials = trials_val;
        }
    }

    let mut map_size = Vec2i::new(51, 51);
    if let Some(x_str) = args.pop() {
        if let Ok(x) = x_str.parse() {
            if let Some(y_str) = args.pop() {
                if let Ok(y) = y_str.parse() {
                    map_size = Vec2i::new(x, y);
                }
            }
        }
    }

    let mut map_type = MapType::DFSRand;
    if let Some(map_type_str) = args.pop() {
        map_type = MapType::parse_map_type_str(map_type_str);
    }

    let mut map_gen = MazeBuilder::new(rng_seed, map_size);
    let mut maps = Vec::new();
    for _ in 0..trials {
        let map = map_type.generate_map(&mut map_gen);
        maps.push(map);
    }

    println!(
        "🤖 Auto Tests\n{} Map, Trials: {}, Map Size: {}, RNG Seed: {}",
        map_type, trials, map_size, rng_seed
    );

    auto_test_behavior(
        "A* Forward",
        rng_seed,
        &maps,
        AStarBehavior::new(true, BreakTieMode::HigherGCost),
    );

    auto_test_behavior(
        "A* Backward",
        rng_seed,
        &maps,
        AStarBehavior::new(false, BreakTieMode::HigherGCost),
    );

    auto_test_behavior(
        "A* Forward Lower G",
        rng_seed,
        &maps,
        AStarBehavior::new(true, BreakTieMode::LowerGCost),
    );

    auto_test_behavior(
        "A* Forward Higher G",
        rng_seed,
        &maps,
        AStarBehavior::new(true, BreakTieMode::HigherGCost),
    );

    auto_test_behavior(
        "Adaptive A*",
        rng_seed,
        &maps,
        AdaptiveAStarBehavior::new(BreakTieMode::HigherGCost, false),
    );
}

fn auto_test_behavior<T: AgentBehavior + Clone + 'static>(
    name: &str,
    rng_seed: u64,
    maps: &Vec<WallMap>,
    agent_behavior: T,
) {
    let mut total_time: f32 = 0.0;
    let mut total_expanded_cell_count: f32 = 0.0;
    for map in maps.iter() {
        let simulation = SimulationBuilder::from_map(
            rng_seed,
            map.clone(),
            true,
            Box::new(agent_behavior.clone()),
        )
        .expect("Expect building to work.");
        let mut instant_runner = SimulationRunner::new_instant(simulation);
        let now = Instant::now();
        {
            instant_runner.run_instant();
        }
        let elapsed_time = now.elapsed();
        total_time += elapsed_time.as_millis() as f32;
        total_expanded_cell_count += instant_runner.simulation.agent.expanded_cell_count as f32;
    }
    let average_time = total_time / maps.len() as f32;
    let average_expanded_cell_count = total_expanded_cell_count / maps.len() as f32;
    println!(
        "{}:\n  Average Time:           {} ms\n  Average Expanded Cells: {} cells",
        name,
        average_time,
        average_expanded_cell_count.separated_string()
    );
}

fn manual_test(mut args: Vec<String>) {
    let mut rng_seed = thread_rng().gen();
    if let Some(seed_str) = args.pop() {
        if let Ok(seed_val) = seed_str.parse() {
            rng_seed = seed_val;
        }
    }

    let mut size = Vec2i::new(25, 25);
    if let Some(x_str) = args.pop() {
        if let Ok(x) = x_str.parse() {
            if let Some(y_str) = args.pop() {
                if let Ok(y) = y_str.parse() {
                    size = Vec2i::new(x, y);
                }
            }
        }
    }
    let mut interval: f32 = 0.25;
    if let Some(interval_str) = args.pop() {
        if let Ok(interval_val) = interval_str.parse() {
            interval = interval_val;
        }
    }

    let mut map_type = MapType::DFSRand;
    if let Some(map_type_str) = args.pop() {
        map_type = MapType::parse_map_type_str(map_type_str);
    }

    let mut reachable_goal = false;
    if let Some(reachable_goal_str) = args.pop() {
        if reachable_goal_str.to_lowercase() == "reachable" {
            reachable_goal = true;
        } else if reachable_goal_str.to_lowercase() == "unreachable" {
            reachable_goal = false;
        }
    }

    let mut break_tie_mode = BreakTieMode::None;
    if let Some(break_tie_mode_str) = args.pop() {
        if break_tie_mode_str.to_lowercase() == "none" {
            break_tie_mode = BreakTieMode::None;
        } else if break_tie_mode_str.to_lowercase() == "high-g" {
            break_tie_mode = BreakTieMode::HigherGCost;
        } else if break_tie_mode_str.to_lowercase() == "low-g" {
            break_tie_mode = BreakTieMode::LowerGCost;
        }
    }

    let mut is_forward = false;
    if let Some(reachable_goal_str) = args.pop() {
        if reachable_goal_str.to_lowercase() == "forward" {
            is_forward = true;
        } else if reachable_goal_str.to_lowercase() == "backward" {
            is_forward = false;
        }
    }

    let mut ai_behavior: Box<dyn AgentBehavior> = Box::new(AStarBehavior {
        break_tie_mode,
        is_forward,
    });
    if let Some(ai_behavior_str) = args.pop() {
        if ai_behavior_str.to_lowercase() == "astar" {
            ai_behavior = Box::new(AStarBehavior::new(is_forward, break_tie_mode));
        } else if ai_behavior_str.to_lowercase() == "adaptive-astar" {
            ai_behavior = Box::new(AdaptiveAStarBehavior::new(break_tie_mode, true));
        }
    }

    let mut map_gen = MazeBuilder::new(rng_seed, size);
    let rand_map = map_type.generate_map(&mut map_gen);
    let simulation =
        SimulationBuilder::from_map(rng_seed, rand_map.clone(), reachable_goal, ai_behavior);
    if let Some(simulation) = simulation {
        let mut runner = SimulationRunner::new(simulation, interval, true);
        runner.run();
    } else {
        println!("⛔ All map cells are connected, cannot set unreachable goal.");
        println!("{}", rand_map);
    }
}
// endregion
