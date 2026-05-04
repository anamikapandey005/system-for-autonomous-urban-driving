import numpy as np
import matplotlib.pyplot as plt

from standalone.world import World
from standalone.sensor import Sensor
from standalone.planner import MultiObjectivePlanner

def main():
    world = World()
    sensor = Sensor(range=40.0, noise_std=0.2)
    planner = MultiObjectivePlanner(target_speed=15.0)

    history_jerk = []
    history_velocity = []
    history_min_dist = []
    
    prev_a = 0.0

    print("Running simulation for metrics collection...")
    for frame in range(300):
        observations = sensor.get_observations(world.ego, world.agents)
        opt_a, opt_steer, _ = planner.plan(world.ego, observations)
        
        # Calculate Jerk (change in acceleration)
        jerk = abs(opt_a - prev_a) / world.dt
        prev_a = opt_a
        
        # Calculate min distance to any agent
        min_dist = float('inf')
        for agent in world.agents:
            dist = np.hypot(agent.x - world.ego.x, agent.y - world.ego.y)
            if dist < min_dist:
                min_dist = dist
                
        history_jerk.append(jerk)
        history_velocity.append(world.ego.v)
        history_min_dist.append(min_dist)
        
        world.step(opt_a, opt_steer)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Multi-Objective Optimization Metrics')

    # Velocity Plot (Efficiency)
    axs[0].plot(history_velocity, color='blue')
    axs[0].axhline(y=15.0, color='r', linestyle='--', label='Target Speed')
    axs[0].set_ylabel('Velocity (m/s)')
    axs[0].set_title('Efficiency: Maintaining Target Speed')
    axs[0].legend()

    # Jerk Plot (Comfort)
    axs[1].plot(history_jerk, color='green')
    axs[1].set_ylabel('Jerk (m/s^3)')
    axs[1].set_title('Comfort: Minimizing Jerk')

    # Safety Plot (Distance to obstacles)
    axs[2].plot(history_min_dist, color='red')
    axs[2].axhline(y=1.5, color='black', linestyle='--', label='Collision Threshold')
    axs[2].set_ylabel('Min Distance (m)')
    axs[2].set_xlabel('Time Steps')
    axs[2].set_title('Safety: Obstacle Avoidance')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('optimization_metrics.png')
    print("Metrics saved to optimization_metrics.png")

if __name__ == '__main__':
    main()
