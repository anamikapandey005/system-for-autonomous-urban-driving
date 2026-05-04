import numpy as np

def calculate_safety_cost(trajectory, obstacles, min_distance=3.0):
    """
    Penalize trajectories that come too close to obstacles.
    Extra penalty for pedestrians.
    """
    cost = 0.0
    for point in trajectory:
        for obs in obstacles:
            if 'position' in obs:
                dist = np.hypot(point[0] - obs['position'][0], point[1] - obs['position'][1])
                
                # Dynamic safety margin based on type
                margin = min_distance
                if obs.get('type') == 'pedestrian':
                    margin += 1.5 # Extra buffer for people
                
                if dist < margin:
                    # Very high penalty for potential collision
                    penalty_scale = 200000.0 if obs.get('type') == 'pedestrian' else 100000.0
                    cost += penalty_scale / (dist + 0.1)
                elif dist < margin * 1.5:
                    cost += 500.0 / (dist - margin + 0.1)
    return cost

def calculate_comfort_cost(trajectory, dt=0.1):
    """
    Penalize high jerk (change in acceleration) and sharp steering.
    """
    cost = 0.0
    if len(trajectory) < 3:
        return 0.0

    for i in range(1, len(trajectory) - 1):
        jerk = (trajectory[i+1][3] - trajectory[i][3]) / dt
        steering_rate = (trajectory[i+1][4] - trajectory[i][4]) / dt
        
        # Penalize steering more to keep the car straighter
        steering_penalty = (trajectory[i][4] ** 2) * 200.0
        
        cost += (jerk ** 2) * 2.0 + (steering_rate ** 2) * 10.0 + steering_penalty
        
    return cost

def calculate_efficiency_cost(trajectory, target_velocity):
    """
    Reward maintaining target speed and penalize unnecessary braking.
    """
    cost = 0.0
    for point in trajectory:
        velocity = point[2]
        cost += abs(target_velocity - velocity) * 2.0
        
        if velocity < 0.1:
            cost += 3000.0 # Heavier penalty for stopping if not needed
        
        acceleration = point[3]
        if acceleration < -3.0: # Hard braking penalty
            cost += abs(acceleration) * 1.0
            
    return cost

def calculate_legal_cost(trajectory, traffic_lights, stop_signs):
    """
    Apply penalties for moving when a red light is active.
    """
    cost = 0.0
    for light in traffic_lights:
        if light['state'] == 'Red':
            for point in trajectory:
                velocity = point[2]
                # Huge penalty for moving at a red light
                if velocity > 0.1:
                    cost += 50000.0 * velocity 
        elif light['state'] == 'Yellow':
            for point in trajectory:
                velocity = point[2]
                # Moderate penalty for moving fast at yellow
                if velocity > 2.0:
                    cost += 5000.0 * velocity
    return cost

def calculate_lane_keeping_cost(trajectory, target_waypoint):
    """
    Penalize distance from the target waypoint.
    Heavier penalty if very far (likely off-road).
    """
    if target_waypoint is None:
        return 0.0
    
    cost = 0.0
    for point in trajectory:
        dist = np.hypot(point[0] - target_waypoint[0], point[1] - target_waypoint[1])
        # Quadratic penalty for lane deviation
        cost += (dist ** 2) * 50.0 + dist * 100.0
    return cost

def evaluate_total_cost(trajectory, obstacles, target_velocity, traffic_rules, target_waypoint=None):
    """
    Combines all costs with weights prioritizing safety and legality.
    """
    # Weights - Safety is top priority
    w_safety = 5000.0   # Increased from 500
    w_legal = 5000.0    # Increased from 500
    w_lane = 1000.0     # Increased from 100
    w_comfort = 10.0    # Increased from 1.0
    w_efficiency = 100.0 # Decreased from 1000.0 to prioritize safety
    
    cost = 0.0
    cost += w_safety * calculate_safety_cost(trajectory, obstacles)
    cost += w_comfort * calculate_comfort_cost(trajectory)
    cost += w_efficiency * calculate_efficiency_cost(trajectory, target_velocity)
    cost += w_legal * calculate_legal_cost(trajectory, traffic_rules['lights'], traffic_rules['signs'])
    cost += w_lane * calculate_lane_keeping_cost(trajectory, target_waypoint)
    
    return cost
