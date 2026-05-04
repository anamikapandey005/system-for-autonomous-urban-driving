import numpy as np
import random
from planning.cost_functions import evaluate_total_cost

class LocalPlanner:
    def __init__(self, target_velocity=30.0):
        self.target_velocity = target_velocity / 3.6 # convert km/h to m/s
        
    def generate_candidate_trajectories(self, current_state, target_waypoint):
        """
        Generate multiple possible trajectories (e.g., using polynomials in Frenet frame).
        For simplicity, we use a basic straight/left/right heuristic sampling here.
        current_state: (x, y, v, yaw)
        """
        trajectories = []
        # A simple trajectory roll-out with finer sampling
        for steer_bias in [-0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for accel_bias in [-8.0, -5.0, -2.0, 0.0, 1.0, 2.0, 4.0, 8.0]:
                traj = []
                x, y, v, yaw = current_state
                steer = steer_bias
                accel = accel_bias
                
                # Simulate 2 seconds ahead at 10Hz
                for _ in range(20):
                    x += v * np.cos(yaw) * 0.1
                    y += v * np.sin(yaw) * 0.1
                    v = max(0.0, v + accel * 0.1)
                    yaw += steer * 0.1
                    traj.append((x, y, v, accel, steer))
                trajectories.append(traj)
                
        return trajectories

    def run_step(self, current_state, target_waypoint, obstacles, traffic_rules):
        """
        1. Generate candidates.
        2. Evaluate multi-objective cost for each.
        3. Return the best trajectory and its immediate control action.
        """
        candidates = self.generate_candidate_trajectories(current_state, target_waypoint)
        
        best_cost = float('inf')
        best_trajectory = None
        
        for traj in candidates:
            cost = evaluate_total_cost(traj, obstacles, self.target_velocity, traffic_rules, target_waypoint)
            if cost < best_cost:
                best_cost = cost
                best_trajectory = traj
                
        if best_trajectory is None:
            # Emergency brake if no valid trajectory
            return None, {'steer': 0.0, 'throttle': 0.0, 'brake': 1.0}
            
        # Extract immediate action from the best trajectory
        optimal_steer = best_trajectory[0][4]
        optimal_accel = best_trajectory[0][3]
        
        throttle = max(0.0, min(1.0, optimal_accel / 5.0))
        brake = max(0.0, min(1.0, -optimal_accel / 5.0))
        steer = max(-1.0, min(1.0, optimal_steer))
        
        control_cmd = {'steer': steer, 'throttle': throttle, 'brake': brake}
        return best_trajectory, control_cmd
