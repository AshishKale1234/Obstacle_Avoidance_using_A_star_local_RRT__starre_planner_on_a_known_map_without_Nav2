# Obstacle Avoidance using A* and Local RRT* Re-planner  
## Navigation on a Known Map (Without Nav2) — ROS 2

This project implements **autonomous navigation and obstacle avoidance on a known map**
using **A\*** for global path planning and a **local RRT\*** re-planner for dynamic obstacle avoidance,
**without using the Nav2 navigation stack**.

Robot localization and goal selection are performed interactively through **RViz** using
*2D Pose Estimate* and *2D Nav Goal*.

---

## Supported ROS 2 Distributions
- **ROS 2 Humble** (Ubuntu 22.04)
- **ROS 2 Jazzy** (Ubuntu 24.04)

---

## System Requirements
- Ubuntu 22.04 (Humble) or Ubuntu 24.04 (Jazzy)
- ROS 2 Desktop installation
- Gazebo
- TurtleBot3 packages
- AMCL and Map Server (for localization on a known map)

---

## Environment Setup
Set the TurtleBot3 model before running:

```bash
export TURTLEBOT3_MODEL=waffle
```

(Optional, persistent):
```bash
echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
```

---

## Build Instructions
After cloning or downloading this repository:

```bash
cd sim_ws_Fall2025
colcon build --symlink-install
source install/local_setup.bash
```

⚠️ The workspace must be sourced in every new terminal.

---

## Running Task 2 — Obstacle Avoidance & Navigation

### 1) Launch simulation, map server, localization, and RViz
```bash
ros2 launch turtlebot3_gazebo navigator.launch.py
```

This launches:
- Gazebo simulation
- Map server (known map)
- AMCL localization
- RViz for user interaction

---

### 2) Run the Task 2 navigation node
Open a **new terminal**, source the workspace, and run:

```bash
cd sim_ws_Fall2025
source install/local_setup.bash
python3 src/turtlebot3_gazebo/src/lab4/task2.py
```

---

### 3) Control via RViz
In RViz:
1. Click **2D Pose Estimate** once to initialize robot localization
2. Click **2D Nav Goal** to send navigation goals

The robot will:
- Plan a global path using **A\***
- Continuously monitor obstacles
- Trigger **local RRT\*** replanning when obstacles obstruct the path
- Safely reach the goal without Nav2

---

## Project Structure
Relevant files:

```
sim_ws_Fall2025/
 └── src/
     └── turtlebot3_gazebo/
         └── src/
             └── lab4/
                 └── task2.py   # A* global planner + local RRT* replanner
```

---

## Key Features
- Known-map navigation
- A* global path planning
- Local RRT* replanning for obstacle avoidance
- RViz-based pose estimation and goal selection
- No Nav2 usage
- Custom planning and control logic

---

## Notes
- Localization is handled using AMCL
- Navigation logic is fully custom (no Nav2 planners/controllers)
- Designed for reproducibility and evaluation

---

## Author
Ashish Kale  
Autonomous Systems — ROS 2
