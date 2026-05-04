import numpy as np

class Sensor:
    def __init__(self, range=60.0, noise_std=0.2):
        self.range = range
        self.noise_std = noise_std
        
    def get_observations(self, ego, agents, traffic_lights, stop_signs, speed_zones=None):
        """
        Observes dynamic agents, traffic lights, stop signs,
        and the current speed limit zone within sensor range.
        """
        obs_agents = []
        for agent in agents:
            dist = np.hypot(agent.x - ego.x, agent.y - ego.y)
            if dist <= self.range:
                obs_agents.append({
                    'id':     agent.id,
                    'x':      agent.x + np.random.normal(0, self.noise_std),
                    'y':      agent.y + np.random.normal(0, self.noise_std),
                    'vx':     agent.vx + np.random.normal(0, self.noise_std * 0.5),
                    'vy':     agent.vy + np.random.normal(0, self.noise_std * 0.5),
                    'radius': agent.radius
                })
                
        obs_lights = []
        for tl in traffic_lights:
            dist = tl.x - ego.x
            if 0 < dist <= self.range:
                obs_lights.append({'x': tl.x, 'y': tl.y, 'state': tl.state})
                
        obs_signs = []
        for sign in stop_signs:
            dist = sign.x - ego.x
            if 0 < dist <= self.range:
                obs_signs.append({'x': sign.x, 'y': sign.y})

        # Current speed limit from speed zones
        current_limit = None
        if speed_zones:
            for zone in speed_zones:
                if zone.x_start <= ego.x < zone.x_end:
                    current_limit = zone.limit_mps
                    break

        return {
            'agents':        obs_agents,
            'lights':        obs_lights,
            'signs':         obs_signs,
            'speed_limit':   current_limit   # m/s, or None
        }
