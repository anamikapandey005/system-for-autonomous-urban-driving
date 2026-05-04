from ultralytics import YOLO
import os

def train_custom_model(data_yaml_path, epochs=50, imgsz=640, batch=16):
    """
    Train a YOLOv8 model on the custom collected CARLA dataset.
    Requires a data.yaml file pointing to the images/labels directories.
    """
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found.")
        print("Please ensure you have generated the dataset and created a data.yaml file.")
        return

    print("Initializing YOLOv8n model...")
    model = YOLO('yolov8n.pt') # Load pre-trained weights to speed up training

    print(f"Starting training for {epochs} epochs...")
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        project='carla_perception',
        name='custom_yolo'
    )
    
    print("Training complete! Model saved to runs/carla_perception/custom_yolo/weights/best.pt")

if __name__ == '__main__':
    # Assuming user creates a data.yaml in the output directory
    yaml_path = r'd:\Tesla\data_collection\_out_data\data.yaml'
    train_custom_model(yaml_path, epochs=30)
