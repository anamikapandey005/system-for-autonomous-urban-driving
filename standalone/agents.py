import numpy as np

class DynamicAgent:
    def __init__(self, id, x, y, vx, vy, radius=1.0):
        self.id = id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

class Pedestrian(DynamicAgent):
    def __init__(self, id, x, y):
        # Pedestrians move randomly but slowly
        vx = np.random.uniform(-1.5, 1.5)
        vy = np.random.uniform(-1.5, 1.5)
        super().__init__(id, x, y, vx, vy, radius=0.5)
        
    def update(self, dt):
        # Slightly erratic movement
        self.vx += np.random.uniform(-0.5, 0.5) * dt
        self.vy += np.random.uniform(-0.5, 0.5) * dt
        self.vx = np.clip(self.vx, -2.0, 2.0)
        self.vy = np.clip(self.vy, -2.0, 2.0)
        super().update(dt)

class ErraticVehicle(DynamicAgent):
    def __init__(self, id, x, y):
        # Erratic vehicles move fast and change lanes suddenly
        vx = np.random.uniform(5.0, 15.0)
        vy = 0.0
        super().__init__(id, x, y, vx, vy, radius=2.0)
        
    def update(self, dt):
        # Sudden lateral movements (lane change proxy)
        if np.random.rand() < 0.05:
            self.vy = np.random.choice([-3.0, 3.0])
        else:
            self.vy *= 0.9 # Dampen lateral movement
            
        super().update(dt)
        # Keep vehicle on the main road
        self.y = np.clip(self.y, -4.0, 4.0)

class CrossTrafficVehicle(DynamicAgent):
    def __init__(self, id, x, y, vy):
        # Moves vertically along the crossroad
        super().__init__(id, x, y, vx=0.0, vy=vy, radius=2.0)
        
    def update(self, dt):
        super().update(dt)
        # Wrap around the crossroad
        if self.y > 30:
            self.y = -30
        elif self.y < -30:
            self.y = 30
