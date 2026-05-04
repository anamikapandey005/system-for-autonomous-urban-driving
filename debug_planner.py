from standalone.world import World
from standalone.sensor import Sensor
from standalone.planner import MultiObjectivePlanner

def run_debug():
    world = World()
    sensor = Sensor(range=60.0, noise_std=0.0) # No noise for debugging
    planner = MultiObjectivePlanner(target_speed=15.0)

    for i in range(50):
        obs = sensor.get_observations(world.ego, world.agents, world.traffic_lights, world.stop_signs)
        opt_a, opt_steer, best_traj, _ = planner.plan(world.ego, obs)
        world.step(opt_a, opt_steer)
        
        agent_strs = [f"A{a.id}(x:{a.x:.1f}, y:{a.y:.1f})" for a in world.agents]
        print(f"Frame {i}: Ego(x:{world.ego.x:.1f}, y:{world.ego.y:.1f}, v:{world.ego.v:.1f}) | a:{opt_a:.1f}, steer:{opt_steer:.2f} | Agents: {', '.join(agent_strs)}")
        if best_traj is None:
            print(">>> EMERGENCY BRAKE TRIGGERED <<<")

if __name__ == "__main__":
    run_debug()
