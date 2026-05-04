import carla
import random
import time
import queue
import numpy as np
import cv2
import os

# Configuration
OUTPUT_DIR = "_out_data"
NUM_FRAMES_TO_COLLECT = 1000

def setup_camera(world, vehicle):
    """Setup an RGB camera attached to the vehicle."""
    bp_library = world.get_blueprint_library()
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    
    # Position the camera at the hood/roof
    transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    return camera

def get_bounding_boxes(world, camera, vehicle):
    """
    Extract 2D bounding boxes for vehicles and pedestrians from the CARLA world.
    (Simplified implementation. True 3D to 2D projection requires camera matrices).
    """
    # Note: For a robust dataset, we would project 3D bounding boxes to 2D image space
    # using the camera intrinsic matrix. This is a placeholder for that logic.
    bounding_boxes = []
    
    # Example logic to get nearby actors
    actors = world.get_actors().filter('vehicle.*')
    for actor in actors:
        if actor.id != vehicle.id:
            distance = actor.get_transform().location.distance(vehicle.get_transform().location)
            if distance < 50.0: # Only consider actors within 50 meters
                # We would project actor.bounding_box to 2D camera plane here
                pass 
                
    return bounding_boxes

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        os.makedirs(os.path.join(OUTPUT_DIR, "images"))
        os.makedirs(os.path.join(OUTPUT_DIR, "labels"))

    actor_list = []
    
    try:
        # Connect to CARLA server
        print("Connecting to CARLA server at localhost:2000...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        # Spawn Ego Vehicle
        bp_library = world.get_blueprint_library()
        vehicle_bp = bp_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f"Spawned ego vehicle: {vehicle.type_id}")

        # Set vehicle to autopilot using Traffic Manager
        vehicle.set_autopilot(True)
        
        # Setup Camera
        camera = setup_camera(world, vehicle)
        actor_list.append(camera)

        # Queue to hold camera data
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        print("Starting data collection...")
        for frame_idx in range(NUM_FRAMES_TO_COLLECT):
            world.tick()
            
            # Retrieve image
            image = image_queue.get()
            
            # Process image array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3] # Remove alpha channel
            
            # Save image
            image_path = os.path.join(OUTPUT_DIR, "images", f"{frame_idx:06d}.jpg")
            cv2.imwrite(image_path, array)
            
            # Retrieve and save bounding box labels (Placeholder)
            # bboxes = get_bounding_boxes(world, camera, vehicle)
            # save_yolo_labels(bboxes, frame_idx)
            
            if frame_idx % 50 == 0:
                print(f"Collected {frame_idx}/{NUM_FRAMES_TO_COLLECT} frames.")

        print("Data collection completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        print("Cleaning up actors...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        for actor in actor_list:
            actor.destroy()
        print("Done.")

if __name__ == '__main__':
    main()
