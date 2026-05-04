import cv2
import torch
from ultralytics import YOLO
import numpy as np

class PerceptionModule:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the perception module with a pre-trained or custom trained YOLOv8 model.
        Using YOLOv8 nano (yolov8n.pt) by default for fast real-time inference.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Perception Model on {self.device}...")
        # Load the model (downloads weights if not present locally)
        self.model = YOLO(model_path)
        
    def process_frame(self, image_array):
        """
        Run object detection on a CARLA camera frame.
        image_array: numpy array of shape (H, W, 3) in BGR format (CARLA default).
        """
        # YOLOv8 (ultralytics) handles BGR numpy arrays natively.
        results = self.model(image_array, verbose=False, device=self.device)[0]
        
        detected_objects = []
        
        for box in results.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]
            
            # We are primarily interested in vehicles, pedestrians, and cyclists
            if class_name in ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']:
                detected_objects.append({
                    'class': class_name,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence
                })
                
        return detected_objects

    def draw_detections(self, image_array, detections):
        """
        Utility to draw bounding boxes on the image for visualization.
        """
        # Image is already in BGR format from CARLA, which cv2 uses.
        img_copy = image_array.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class']} {det['confidence']:.2f}"
            
            color = (0, 255, 0) if det['class'] == 'car' else (0, 0, 255)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return img_copy
