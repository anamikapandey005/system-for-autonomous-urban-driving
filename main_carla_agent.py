import carla
import time
import numpy as np
import cv2
import queue

from perception.carla_inference import PerceptionModule
from planning.local_planner import LocalPlanner
from control.mpc_controller import MPCController
from data_collection.collect_data import setup_camera

def main():
    actor_list = []
    
    try:
        # 1. Setup CARLA Client
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # 2. Spawn Ego Vehicle
        bp_library = world.get_blueprint_library()
        vehicle_bp = bp_library.filter('vehicle.tesla.model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print("Spawned ego vehicle.")
        
        # 3. Setup Sensors (Camera)
        camera = setup_camera(world, vehicle)
        actor_list.append(camera)
        
        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        
        # 4. Initialize Modular Stack
        perception = PerceptionModule(model_path='yolov8n.pt') # Using basic YOLO for now
        planner = LocalPlanner(target_velocity=30.0)
        controller = MPCController(vehicle)
        
        print("Starting Autonomous Loop...")
        
        # 5. Main Control Loop
        while True:
            world.tick()
            
            # --- PERCEPTION ---
            # Get Image from Queue
            image = image_queue.get()
            img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img_array = np.reshape(img_array, (image.height, image.width, 4))
            img_array = img_array[:, :, :3]
            
            # Run YOLO Detection
            detections = perception.process_frame(img_array)
            
            # --- PLANNING ---
            # Get Current State
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            v_mag = np.hypot(velocity.x, velocity.y)
            yaw = np.radians(transform.rotation.yaw)
            
            current_state = (transform.location.x, transform.location.y, v_mag, yaw)
            target_waypoint = None # Simplified for local planning only
            
            # Formulate obstacles dictionary from detections
            obstacles = [] # TODO: map from 2D pixel to 3D world
            traffic_rules = {'lights': [], 'signs': []}
            
            # Generate best trajectory and control command
            best_traj, control_cmd = planner.run_step(current_state, target_waypoint, obstacles, traffic_rules)
            
            # --- CONTROL ---
            controller.run_step(control_cmd)
            
            # Visualization
            vis_image = perception.draw_detections(img_array, detections)
            cv2.imshow("Ego Vehicle View", vis_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Simulation ended: {e}")
        
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
        for actor in actor_list:
            actor.destroy()

if __name__ == '__main__':
    main()
