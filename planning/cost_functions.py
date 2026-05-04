import numpy as np

def calculate_safety_cost(trajectory, obstacles, min_distance=3.0):
    """
    Penalize trajectories that come too close to obstacles.
    trajectory: List of (x, y) waypoints.
    obstacles: List of dicts representing detected obstacles (bounding boxes or positions).
    """
    cost = 0.0
    for point in trajectory:
        for obs in obstacles:
            # Assuming obs has a 'position' (x,y) from the sensor fusion
            # In a real implementation, we map 2D bounding boxes to 3D world coordinates
            if 'position' in obs:
                dist = np.hypot(point[0] - obs['position'][0], point[1] - obs['position'][1])
                if dist < min_distance:
                    # Infinite/extremely high cost for collision
                    return float('inf')
                elif dist < min_distance * 2:
                    cost += 100.0 / (dist - min_distance + 0.1)
    return cost

def calculate_comfort_cost(trajectory, dt=0.1):
    """
    Penalize high jerk (change in acceleration) and sharp steering.
    trajectory: List of (x, y, velocity, acceleration, steering_angle).
    """
    cost = 0.0
    if len(trajectory) < 3:
        return 0.0

    for i in range(1, len(trajectory) - 1):
        # Jerk is the derivative of acceleration
        jerk = (trajectory[i+1][3] - trajectory[i][3]) / dt
        steering_rate = (trajectory[i+1][4] - trajectory[i][4]) / dt
        
        # Penalize absolute steering to keep the car straight when not avoiding obstacles
        steering_penalty = abs(trajectory[i][4]) * 50.0
        
        cost += (jerk ** 2) * 1.5 + (steering_rate ** 2) * 2.0 + steering_penalty
        
    return cost

def calculate_efficiency_cost(trajectory, target_velocity):
    """
    Reward maintaining target speed and penalize unnecessary braking.
    """
    cost = 0.0
    for point in trajectory:
        velocity = point[2]
        cost += abs(target_velocity - velocity) * 1.0
        
        # Penalize hard braking unless necessary for safety (which is handled in safety cost)
        acceleration = point[3]
        if acceleration < -2.0:
            cost += abs(acceleration) * 0.5
            
    return cost

def calculate_legal_cost(trajectory, traffic_lights, stop_signs):
    """
    Apply penalties for breaking traffic rules.
    """
    cost = 0.0
    # In CARLA, we can query traffic light states directly.
    # If the trajectory passes a red light stop line, add massive penalty.
    # Placeholder logic
    return cost

def evaluate_total_cost(trajectory, obstacles, target_velocity, traffic_rules):
    """
    Combines all costs with specific weights to balance the conflicting objectives.
    """
    w_safety = 1000.0
    w_comfort = 10.0
    w_efficiency = 5.0
    w_legal = 500.0
    
    cost = 0.0
    cost += w_safety * calculate_safety_cost(trajectory, obstacles)
    cost += w_comfort * calculate_comfort_cost(trajectory)
    cost += w_efficiency * calculate_efficiency_cost(trajectory, target_velocity)
    cost += w_legal * calculate_legal_cost(trajectory, traffic_rules['lights'], traffic_rules['signs'])
    
    return cost
