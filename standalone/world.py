import numpy as np
from standalone.kinematics import BicycleModel
from standalone.agents import Pedestrian, ErraticVehicle, CrossTrafficVehicle

class TrafficLight:
    def __init__(self, x, y, cycle_length=15.0):
        self.x = x
        self.y = y
        self.state = 'GREEN'
        self.timer = 0.0
        self.cycle_length = cycle_length
        
    def update(self, dt):
        self.timer += dt
        if self.timer > self.cycle_length:
            self.timer = 0.0
            if self.state == 'GREEN':
                self.state = 'YELLOW'
            elif self.state == 'YELLOW':
                self.state = 'RED'
            else:
                self.state = 'GREEN'

class StopSign:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SpeedZone:
    """A zone of road with a defined speed limit (m/s)."""
    def __init__(self, x_start, x_end, limit_mps, label=''):
        self.x_start = x_start
        self.x_end = x_end
        self.limit_mps = limit_mps
        self.label = label

class World:
    def __init__(self):
        self.ego = BicycleModel(x=0, y=0, yaw=0, v=10.0)
        self.agents = []
        self.traffic_lights = []
        self.stop_signs = []
        self.speed_zones = []
        self.time = 0.0
        self.dt = 0.1

        # --- Metrics ---
        self.travel_distance = 0.0
        self.violations = []            # list of {time, type}
        self.red_light_violations = 0
        self.stop_sign_violations = 0
        self.speeding_ticks = 0         # count of 0.1s ticks while speeding
        self._prev_x = 0.0
        self._stopped_at_sign = set()   # signs the ego properly stopped at

        self.setup_infrastructure()
        self.spawn_agents()
        
    def setup_infrastructure(self):
        # Two traffic lights
        self.traffic_lights.append(TrafficLight(60,  0, cycle_length=12.0))
        self.traffic_lights.append(TrafficLight(130, 0, cycle_length=18.0))
        # Stop sign
        self.stop_signs.append(StopSign(100, 0))
        # Speed zones: (x_start, x_end, limit m/s, label)
        self.speed_zones.append(SpeedZone(0,   50,  15.0, '54 km/h'))
        self.speed_zones.append(SpeedZone(50,  90,   8.3, '30 km/h'))
        self.speed_zones.append(SpeedZone(90,  140, 13.9, '50 km/h'))
        self.speed_zones.append(SpeedZone(140, 9999,20.0, '72 km/h'))
        
    def spawn_agents(self):
        self.agents.append(Pedestrian(1, 30, -5))
        self.agents.append(Pedestrian(2, 45,  5))
        self.agents.append(ErraticVehicle(3, 50, 0))
        self.agents.append(CrossTrafficVehicle(4, 60, 15, -6.0))

    def get_current_speed_limit(self):
        """Return active SpeedZone for ego's current x position."""
        for zone in self.speed_zones:
            if zone.x_start <= self.ego.x < zone.x_end:
                return zone
        return None

    def check_violations(self):
        """Detect traffic law violations this timestep."""
        ego = self.ego
        t = round(self.time, 1)

        # 1. Speed limit violation
        zone = self.get_current_speed_limit()
        if zone and ego.v > zone.limit_mps + 0.5:
            self.speeding_ticks += 1
            # Log once per second (every 10 ticks)
            if self.speeding_ticks % 10 == 1:
                self.violations.append({
                    'time': t,
                    'type': f'SPEEDING  {ego.v*3.6:.0f} > {zone.limit_mps*3.6:.0f} km/h'
                })
        else:
            self.speeding_ticks = 0

        # 2. Red / Yellow light violation (crossed stop-line while not green)
        for tl in self.traffic_lights:
            if tl.state in ['RED', 'YELLOW']:
                if self._prev_x < tl.x <= ego.x and ego.v > 0.5:
                    self.red_light_violations += 1
                    self.violations.append({'time': t, 'type': f'RED LIGHT  x={tl.x:.0f}'})

        # 3. Stop sign violation (must stop < 0.5 m/s before crossing line)
        for sign in self.stop_signs:
            sid = id(sign)
            dist = sign.x - ego.x
            # Properly stopped before the sign
            if 0 < dist <= 3.0 and ego.v < 0.5:
                self._stopped_at_sign.add(sid)
            # Crossed the line
            if self._prev_x < sign.x <= ego.x:
                if sid not in self._stopped_at_sign:
                    self.stop_sign_violations += 1
                    self.violations.append({'time': t, 'type': f'STOP SIGN  x={sign.x:.0f}'})
                self._stopped_at_sign.add(sid)

        self._prev_x = ego.x

    def step(self, ego_a, ego_delta):
        self._prev_x = self.ego.x
        self.time += self.dt
        self.travel_distance += self.ego.v * self.dt

        # Update ego
        self.ego.update(ego_a, ego_delta, self.dt)
        
        # Check violations
        self.check_violations()
        
        # Update dynamic agents with wrap-around
        for agent in self.agents:
            agent.update(self.dt)
            if agent.x - self.ego.x < -20:
                agent.x += 160
            elif agent.x - self.ego.x > 160:
                agent.x -= 160
                
        # Update traffic lights
        for tl in self.traffic_lights:
            tl.update(self.dt)
            if tl.x - self.ego.x < -10:
                tl.x += 160

        # Update stop signs (wrap)
        for sign in self.stop_signs:
            if sign.x - self.ego.x < -10:
                sign.x += 160
                self._stopped_at_sign.discard(id(sign))
