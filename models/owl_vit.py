import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from utils.base_model import BaseOVDetector


class OWLViT(BaseOVDetector):
    def __init__(self, device: str = "cuda"):
        super().__init__("OWL-ViT", device)
        self.processor = None
        
    def load_model(self, checkpoint_path: str = None):
        model_name = "google/owlvit-base-patch32"
        if checkpoint_path:
            self.model = OwlViTForObjectDetection.from_pretrained(checkpoint_path).to(self.device)
            self.processor = OwlViTProcessor.from_pretrained(checkpoint_path)
        else:
            self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
            self.processor = OwlViTProcessor.from_pretrained(model_name)
        
        self.model.eval()
        
    def preprocess(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        return image
    
    def postprocess(self, outputs: Any, image_shape: Tuple[int, int], target_sizes: torch.Tensor) -> Dict[str, Any]:
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.0
        )
        
        detections = []
        for result in results:
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                detections.append({
                    'bbox': box.tolist(),
                    'score': float(score),
                    'class_id': int(label)
                })
        
        return {'detections': detections}
    
    def detect(self, image: np.ndarray, text_queries: List[str], confidence_threshold: float = 0.3) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.processor(
            text=text_queries,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
        results = self.postprocess(outputs, image.shape[:2], target_sizes)
        
        filtered_detections = []
        for detection in results['detections']:
            if detection['score'] >= confidence_threshold:
                if detection['class_id'] < len(text_queries):
                    detection['class_name'] = text_queries[detection['class_id']]
                else:
                    detection['class_name'] = 'unknown'
                filtered_detections.append(detection)
        
        return {'detections': filtered_detections}