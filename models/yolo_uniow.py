import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from utils.base_model import BaseOVDetector
from transformers import CLIPProcessor, CLIPModel
from ultralytics.nn.modules import Detect
import cv2


class YOLOUniOW(BaseOVDetector):
    def __init__(self, device: str = "cuda"):
        super().__init__("YOLO-UniOW", device)
        self.clip_processor = None
        self.clip_model = None
        
    def load_model(self, checkpoint_path: str = None):
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        from ultralytics.nn.tasks import DetectionModel
        self.model = DetectionModel('yolov8s.yaml', ch=3, nc=80).to(self.device)
        
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        h, w = image.shape[:2]
        new_shape = (640, 640)
        
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw // 2, dh // 2
        
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = dh, dh
        left, right = dw, dw
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float() / 255.0
        image = image.unsqueeze(0).to(self.device)
        
        return image
    
    def get_text_features(self, text_queries: List[str]) -> torch.Tensor:
        text_inputs = self.clip_processor(text=text_queries, return_tensors="pt", padding=True).to(self.device)
        text_features = self.clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def postprocess(self, outputs: Any, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        predictions = outputs[0]
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        
        scale_x = image_shape[1] / 640
        scale_y = image_shape[0] / 640
        
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        results = []
        for i in range(len(boxes)):
            if scores[i] > 0.01:
                results.append({
                    'bbox': boxes[i].cpu().numpy().tolist(),
                    'score': float(scores[i].cpu().numpy()),
                    'features': predictions[i, 5:].cpu().numpy()
                })
        
        return {'detections': results}
    
    def detect(self, image: np.ndarray, text_queries: List[str], confidence_threshold: float = 0.3) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        preprocessed_image = self.preprocess(image)
        text_features = self.get_text_features(text_queries)
        
        with torch.no_grad():
            outputs = self.model(preprocessed_image)
        
        results = []
        for output in outputs:
            if len(output.shape) == 3:
                output = output[0]
            
            for detection in output:
                box = detection[:4].cpu().numpy()
                obj_score = detection[4].cpu().numpy()
                
                if obj_score < confidence_threshold:
                    continue
                
                region = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if region.size == 0:
                    continue
                
                region_inputs = self.clip_processor(images=region, return_tensors="pt").to(self.device)
                region_features = self.clip_model.get_image_features(**region_inputs)
                region_features = region_features / region_features.norm(dim=-1, keepdim=True)
                
                similarities = (region_features @ text_features.T).squeeze(0)
                max_sim_idx = similarities.argmax().item()
                max_sim = similarities[max_sim_idx].item()
                
                if max_sim * obj_score > confidence_threshold:
                    results.append({
                        'bbox': box.tolist(),
                        'score': float(max_sim * obj_score),
                        'class_id': max_sim_idx,
                        'class_name': text_queries[max_sim_idx]
                    })
        
        return {'detections': results}