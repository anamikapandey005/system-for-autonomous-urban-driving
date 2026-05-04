import numpy as np

class BicycleModel:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, L=2.5, max_steer=np.radians(30), max_a=3.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L  # Wheelbase
        self.max_steer = max_steer
        self.max_a = max_a
        
    def update(self, a, delta, dt):
        """
        Update the state of the vehicle using Kinematic Bicycle Model.
        a: acceleration (m/s^2)
        delta: steering angle (rad)
        dt: time step
        """
        # Constrain inputs
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        a = np.clip(a, -self.max_a, self.max_a)
        
        # Update state
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += (self.v / self.L) * np.tan(delta) * dt
        self.v += a * dt
        self.v = max(0.0, self.v)  # No reversing
        
        # Keep yaw within [-pi, pi]
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi
        
        return self.x, self.y, self.yaw, self.v
