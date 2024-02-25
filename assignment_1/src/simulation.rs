use owo_colors::OwoColorize;
use std::{fmt::Display, thread::sleep, time::Duration};

use crate::prelude::*;

// region Simulation
#[derive(Debug)]
pub struct Simulation {
    pub wall_map: WallMap,
    pub connected_components_map: ConnectedComponentsMap,
    pub agent: Agent,
    pub prev_agent_pos: Vec2i,
    pub steps: i32,
    pub result: Option<bool>,
}
impl Simulation {
    pub fn new(map: WallMap, mut agent: Agent) -> Simulation {
        let connected_components_map = ConnectedComponentsMap::from_wall_map(&map);
        agent.init(map.size().clone());
        Simulation {
            wall_map: map,
            connected_components_map,
            prev_agent_pos: agent.position,
            agent,
            steps: 0,
            result: None,
        }
    }
    pub fn is_running(&self) -> bool {
        self.result.is_none()
    }
    pub fn is_complete(&self) -> bool {
        self.result.is_some()
    }
    pub fn step(&mut self) {
        self.prev_agent_pos = self.agent.position;
        if self.is_complete() {
            return;
        }
        self.agent.step(&self.wall_map);
        self.steps += 1;
        if let AgentStatus::Complete(result) = self.agent.status {
            self.result = Some(result);
        }
    }
    pub fn simulation_str(&self) -> String {
        let mut str = String::new();
        str += &format!(
            "Simulation({}, {})\n Step {}\n  Expanded Cells: {} cells\n  Step Expanded Cells: {} cells\n",
            self.wall_map.size().x, self.wall_map.size().y, self.steps, self.agent.expanded_cell_count, self.agent.step_expanded_cell_count
        );
        str += &self
            .agent
            .mind_map_str(self.prev_agent_pos)
            .join_sides_spaced(&self.full_map_str());
        str
    }
    fn get_connected_component_string(&self, pos: Vec2i) -> String {
        if let Ok(comp_id) = self.connected_components_map.get_cell(pos) {
            match comp_id % 7 {
                0 => format!("{}", " 0".truecolor(168, 41, 32)),
                1 => format!("{}", " 1".truecolor(168, 89, 32)),
                2 => format!("{}", " 2".truecolor(168, 161, 32)),
                3 => format!("{}", " 3".truecolor(105, 168, 32)),
                4 => format!("{}", " 4".truecolor(32, 168, 154)),
                5 => format!("{}", " 5".truecolor(114, 32, 168)),
                6 => format!("{}", " 6".truecolor(168, 32, 123)),
                _ => "  ".to_owned(),
            }
        } else {
            "  ".to_owned()
        }
    }
    pub fn full_map_str(&self) -> String {
        let mut str = String::new();
        str += &format!(
            "{}\n",
            "XX".repeat(self.wall_map.size().x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        for y in 0..self.wall_map.size().y {
            str += &format!("{}", "XX".on_truecolor(125, 125, 125));
            for x in 0..self.wall_map.size().x {
                let pos = Vec2i::new(x, y);
                if self.wall_map.get_cell(pos).unwrap() {
                    str += &format!("{}", "XX".on_truecolor(125, 125, 125));
                } else {
                    if self.prev_agent_pos == pos {
                        str += &format!("{}", "AA".on_truecolor(69, 163, 65));
                    } else if self.agent.goal == pos {
                        str += &format!(
                            "{}",
                            "GG".truecolor(240, 227, 46).on_truecolor(230, 137, 39)
                        );
                    } else {
                        str += &self.get_connected_component_string(pos);
                    }
                }
            }
            str += &format!("{}\n", "XX".on_truecolor(125, 125, 125));
        }
        str += &format!(
            "{}\n",
            "XX".repeat(self.wall_map.size().x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        str
    }
}
impl Display for Simulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.simulation_str())?;
        Ok(())
    }
}
// endregion

// region SimulationBuilder
pub struct SimulationBuilder;
impl SimulationBuilder {
    pub fn from_text(mut text: &str, agent_behavior: Box<dyn AgentBehavior>) -> Option<Simulation> {
        text = text.trim_matches('\n');
        let lines: Vec<&str> = text.lines().collect();
        let y = lines.len();
        let last = lines.last();
        if last.is_none() {
            return None;
        }

        let x = last.unwrap().chars().count();
        let size = Vec2i::new(x as i32, y as i32);

        let mut map: WallMap = WallMap::new(size);
        let mut goal_pos: Option<Vec2i> = None;
        let mut agent_pos: Option<Vec2i> = None;

        let mut y = 0;
        for line in lines {
            if y >= size.y {
                return None;
            }
            let mut x = 0;
            for char in line.chars() {
                if x >= size.x {
                    return None;
                }
                let pos = Vec2i::new(x, y);
                if char == '#' {
                    map.set_cell(pos, true).unwrap();
                } else if char == 'G' {
                    if goal_pos.is_some() {
                        return None;
                    }
                    goal_pos = Some(pos);
                } else if char == 'A' {
                    if agent_pos.is_some() {
                        return None;
                    }
                    agent_pos = Some(pos);
                }
                x += 1;
            }
            y += 1;
        }

        if goal_pos.is_none() || agent_pos.is_none() {
            return None;
        }
        let agent = Agent::new(agent_pos.unwrap(), goal_pos.unwrap(), agent_behavior);
        Some(Simulation::new(map, agent))
    }
    pub fn from_map(
        rng_seed: u64,
        map: WallMap,
        set_reachable_goal: bool,
        agent_behavior: Box<dyn AgentBehavior>,
    ) -> Option<Simulation> {
        let goal = {
            if set_reachable_goal {
                map.find_longest_reachable_goal(Vec2i::ZERO)
            } else {
                if let Some(unreachable_goal) = map.find_unreachable_goal(rng_seed, Vec2i::ZERO) {
                    unreachable_goal
                } else {
                    return None;
                }
            }
        };
        let agent = Agent::new(Vec2i::ZERO, goal, agent_behavior);
        Some(Simulation::new(map, agent))
    }
}
// endregion

// region SimulationRunner
pub struct SimulationRunner {
    pub simulation: Simulation,
    pub interval: f32,
    pub print: bool,
}
impl SimulationRunner {
    pub fn new_instant(simulation: Simulation) -> SimulationRunner {
        SimulationRunner::new(simulation, 0.0, false)
    }
    pub fn new(simulation: Simulation, interval: f32, print: bool) -> SimulationRunner {
        SimulationRunner {
            simulation,
            interval,
            print,
        }
    }
    pub fn run(&mut self) -> bool {
        if self.is_instant() {
            return self.run_instant();
        }

        if self.print && self.interval > 0.0 {
            clearscreen::clear().unwrap();
            println!("🚀 Simulation Start");
            println!("{}", self.simulation);
        }
        while self.simulation.is_running() {
            self.simulation.step();
            if self.print && self.interval > 0.0 {
                clearscreen::clear().unwrap();
                println!("{}", self.simulation);
            }
            if self.interval > 0.0 {
                sleep(Duration::from_secs_f32(self.interval))
            }
        }
        let result = self.simulation.result.unwrap();
        if self.print {
            clearscreen::clear().unwrap();
            self.simulation.step();
            println!("{}", self.simulation);
            println!(
                "🛑 Simulation End.\n  Goal Reached? {}\n  Expanded Cells: {} cells",
                result, self.simulation.agent.expanded_cell_count
            );
        }
        result
    }
    pub fn run_instant(&mut self) -> bool {
        while self.simulation.is_running() {
            self.simulation.step();
        }
        self.simulation.result.unwrap()
    }
    pub fn is_instant(&self) -> bool {
        !self.print && self.interval <= 0.0
    }
}
// endregion
