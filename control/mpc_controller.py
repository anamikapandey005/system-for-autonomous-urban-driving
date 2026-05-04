import carla

class MPCController:
    """
    A simplified controller wrapper. In a full production system, this would be a Model Predictive Controller
    that solves a constrained optimization problem. For this prototype, we translate the high-level
    acceleration/steering commands from the Local Planner into CARLA VehicleControls.
    """
    def __init__(self, vehicle):
        self.vehicle = vehicle
        
    def run_step(self, control_cmd):
        """
        Applies control command to the CARLA vehicle.
        control_cmd: dict with 'steer', 'throttle', 'brake'
        """
        carla_control = carla.VehicleControl()
        carla_control.steer = control_cmd.get('steer', 0.0)
        carla_control.throttle = control_cmd.get('throttle', 0.0)
        carla_control.brake = control_cmd.get('brake', 0.0)
        carla_control.hand_brake = False
        carla_control.manual_gear_shift = False
        
        self.vehicle.apply_control(carla_control)
        return carla_control
