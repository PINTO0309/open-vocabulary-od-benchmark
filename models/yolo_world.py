import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
from utils.base_model import BaseOVDetector


class YOLOWorld(BaseOVDetector):
    def __init__(self, device: str = "cuda"):
        super().__init__("YOLO-World", device)
        
    def load_model(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            self.model = YOLO("yolov8s-worldv2.pt")
        else:
            self.model = YOLO(checkpoint_path)
            
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image
    
    def postprocess(self, outputs: Any, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        results = []
        for output in outputs:
            if output.boxes is not None:
                boxes = output.boxes.xyxy.cpu().numpy()
                scores = output.boxes.conf.cpu().numpy()
                classes = output.boxes.cls.cpu().numpy()
                
                for box, score, cls in zip(boxes, scores, classes):
                    results.append({
                        'bbox': box.tolist(),
                        'score': float(score),
                        'class_id': int(cls)
                    })
        
        return {'detections': results}
    
    def detect(self, image: np.ndarray, text_queries: List[str], confidence_threshold: float = 0.3) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.model.set_classes(text_queries)
        
        results = self.model.predict(
            image, 
            conf=confidence_threshold,
            device=self.device.type
        )
        
        processed_results = self.postprocess(results, image.shape[:2])
        
        for detection in processed_results['detections']:
            if detection['class_id'] < len(text_queries):
                detection['class_name'] = text_queries[detection['class_id']]
            else:
                detection['class_name'] = 'unknown'
        
        return processed_results