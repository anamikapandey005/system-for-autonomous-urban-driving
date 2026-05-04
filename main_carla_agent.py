import carla
import time
import numpy as np
import cv2
import queue
import random

from perception.carla_inference import PerceptionModule
from planning.local_planner import LocalPlanner
from control.mpc_controller import MPCController
from data_collection.collect_data import setup_camera

def main():
    actor_list = []
    
    try:
        # 1. Setup CARLA Client
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)
        world = client.get_world()
        
        # 2. Spawn Ego Vehicle
        bp_library = world.get_blueprint_library()
        vehicle_bp = bp_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f"Spawned ego vehicle at {spawn_point.location}")
        
        # 3. Setup Sensors (Camera)
        camera = setup_camera(world, vehicle)
        actor_list.append(camera)
        
        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        
        # 4. Initialize Modular Stack
        perception = PerceptionModule(model_path='yolov8n.pt') 
        planner = LocalPlanner(target_velocity=30.0) # Reduced from 50 to 30 km/h for safety
        controller = MPCController(vehicle)
        
        print("Starting Autonomous Loop...")
        
        # 5. Main Control Loop
        while True:
            world.tick()
            
            # --- PERCEPTION ---
            # Get Image from Queue
            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("No image received from camera sensor.")
                continue
                
            img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img_array = np.reshape(img_array, (image.height, image.width, 4))
            img_array = img_array[:, :, :3]
            
            # Run YOLO Detection (for visualization)
            detections = perception.process_frame(img_array)
            
            # --- PLANNING ---
            # Get Current State
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            v_mag = np.hypot(velocity.x, velocity.y)
            yaw = np.radians(transform.rotation.yaw)
            
            current_state = (transform.location.x, transform.location.y, v_mag, yaw)
            
            # Use shorter lookahead for better precision in urban environments
            carla_map = world.get_map()
            waypoint = carla_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            
            # Lookahead distance: 8.0m for better cornering
            lookahead_dist = 8.0
            next_wps = waypoint.next(lookahead_dist)
            next_waypoint = next_wps[0] if next_wps else waypoint
            
            target_waypoint = (next_waypoint.transform.location.x, next_waypoint.transform.location.y)
            
            # Formulate obstacles list
            obstacles = []
            ego_forward = transform.get_forward_vector()
            
            # Detect vehicles
            for actor in world.get_actors().filter('vehicle.*'):
                if actor.id != vehicle.id:
                    loc = actor.get_location()
                    dist = transform.location.distance(loc)
                    if dist < 40.0:
                        vec_to_obs = loc - transform.location
                        dot = ego_forward.x * vec_to_obs.x + ego_forward.y * vec_to_obs.y
                        if dot > 0:
                            obstacles.append({'position': (loc.x, loc.y), 'id': actor.id, 'type': 'vehicle'})

            # Detect pedestrians
            for actor in world.get_actors().filter('walker.*'):
                loc = actor.get_location()
                dist = transform.location.distance(loc)
                if dist < 30.0:
                    obstacles.append({'position': (loc.x, loc.y), 'id': actor.id, 'type': 'pedestrian'})
            
            traffic_rules = {'lights': [], 'signs': []}
            # Check for traffic light state
            if vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_rules['lights'].append({'state': 'Red'})
                elif traffic_light.get_state() == carla.TrafficLightState.Yellow:
                    traffic_rules['lights'].append({'state': 'Yellow'})
            
            # Generate best trajectory and control command
            best_traj, control_cmd = planner.run_step(current_state, target_waypoint, obstacles, traffic_rules)
            
            # --- CONTROL ---
            controller.run_step(control_cmd)
            
            min_dist = min([transform.location.distance(carla.Location(o['position'][0], o['position'][1], transform.location.z)) for o in obstacles]) if obstacles else 999
            print(f"Speed: {v_mag*3.6:.1f} km/h | Dist: {min_dist:.1f}m | T: {control_cmd['throttle']:.2f} | B: {control_cmd['brake']:.2f} | S: {control_cmd['steer']:.2f}")
            
            # Visualization
            vis_image = perception.draw_detections(img_array, detections)
            cv2.imshow("Ego Vehicle View", vis_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Simulation ended: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
        for actor in actor_list:
            actor.destroy()

if __name__ == '__main__':
    main()
