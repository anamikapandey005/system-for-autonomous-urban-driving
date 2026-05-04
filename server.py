import threading
import time
import json
from flask import Flask, jsonify, send_from_directory

from standalone.world import World
from standalone.sensor import Sensor
from standalone.planner import MultiObjectivePlanner

app = Flask(__name__, static_folder='web')

# Global Simulation State
world = World()
sensor = Sensor(range=60.0, noise_std=0.1)
planner = MultiObjectivePlanner(target_speed=15.0)

current_state = {}

def simulation_loop():
    global current_state
    while True:
        # Perception
        observations = sensor.get_observations(world.ego, world.agents, world.traffic_lights, world.stop_signs)
        
        # Planning
        opt_a, opt_steer, best_traj, predictions = planner.plan(world.ego, observations)
        
        # Physics Update
        world.step(opt_a, opt_steer)
        
        # Prepare state for frontend
        state = {
            'ego': {
                'x': world.ego.x,
                'y': world.ego.y,
                'yaw': world.ego.yaw,
                'v': world.ego.v,
                'a': opt_a,
                'steer': opt_steer
            },
            'agents': [{'id': a.id, 'x': a.x, 'y': a.y, 'radius': a.radius, 'type': 'pedestrian' if 'Pedestrian' in str(type(a)) else 'vehicle'} for a in world.agents],
            'traffic_lights': [{'x': tl.x, 'y': tl.y, 'state': tl.state} for tl in world.traffic_lights],
            'stop_signs': [{'x': s.x, 'y': s.y} for s in world.stop_signs],
            'trajectory': best_traj if best_traj else [],
            'predictions': predictions
        }
        current_state = state
        time.sleep(0.05) # 20 FPS

@app.route('/')
def serve_index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

@app.route('/api/state')
def get_state():
    return jsonify(current_state)

if __name__ == '__main__':
    # Start simulation thread
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    print("Starting Premium Web Dashboard on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
