# Obstacle Avoidance using A* and Local RRT* Re-planning

- Autonomous navigation on a **known indoor map** using TurtleBot3 and ROS 2
- Global path planning implemented using **A\*** (no Nav2 planners)
- **Local RRT\*** re-planner triggered for obstacle avoidance
- Robot localization performed using **AMCL**
- User interaction through **RViz**:
  - *2D Pose Estimate* for initial localization
  - *2D Nav Goal* for goal selection
- Fully custom navigation and control pipeline
- No use of the **Nav2 navigation stack**
- Designed and evaluated in **Gazebo simulation**
- Self-contained ROS 2 workspace

Detailed build and run instructions are provided inside  
**`sim_ws_Fall2025/README.md`**
