# Autonomous Urban Driving System

A complete standalone 2D Python simulation of an autonomous vehicle that optimizes multiple conflicting objectives (Safety, Comfort, and Efficiency) in a highly dynamic environment.

## Overview
This project implements a Multi-Objective trajectory rollout planner on top of a Kinematic Bicycle Model. It simulates a dynamic urban environment with erratic pedestrians and vehicles, partial sensor observability (Gaussian noise), and complex decision-making.

### Key Objectives Handled:
1. **Safety**: Hard constraints on collision using inverse-distance penalties.
2. **Comfort**: Minimization of Jerk (derivative of acceleration) and sharp steering angles.
3. **Efficiency**: Maintaining a target travel speed while avoiding harsh braking.

## Project Structure
- `standalone/world.py`: Manages the 2D coordinate space and time updates.
- `standalone/kinematics.py`: The Ego vehicle physics (Bicycle Model).
- `standalone/agents.py`: Dynamic obstacle classes (Pedestrians, Erratic Drivers).
- `standalone/sensor.py`: Simulates limited range and noisy perception.
- `standalone/planner.py`: Multi-objective trajectory generator and cost evaluator.
- `standalone/run_simulation.py`: The main entry point featuring a Matplotlib live animation.

## How to Run
Ensure you have `numpy` and `matplotlib` installed:
```bash
pip install numpy matplotlib
```

Start the interactive simulation window:
```bash
python -m standalone.run_simulation
```

## How It Works
The planner generates dozens of candidate trajectories into the future. Each trajectory is scored against the multi-objective cost function. The trajectory with the lowest combined cost (representing the best trade-off between safety, comfort, and speed) is selected, and its first control action is applied to the car's physical kinematics.

## Performance Metrics
Run the following command to execute the simulation headlessly and generate optimization graphs:
```bash
python -m standalone.generate_metrics
```
This will output an `optimization_metrics.png` file plotting:
1. **Efficiency**: How well the car maintains the target speed.
2. **Comfort**: Spikes in Jerk (which are minimized).
3. **Safety**: Distance to obstacles, proving collisions are avoided.

![Optimization Metrics](file:///d:/Tesla/optimization_metrics.png)
