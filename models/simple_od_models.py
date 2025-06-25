import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from utils.base_model import BaseOVDetector
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer
import cv2


class SimpleYOLOBasedOVDetector(BaseOVDetector):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        self.text_encoder = None
        self.text_processor = None
        
    def load_model(self, checkpoint_path: str = None):
        self.text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        from ultralytics.nn.tasks import DetectionModel
        self.model = DetectionModel('yolov8n.yaml', ch=3, nc=80).to(self.device)
        
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()
        self.text_encoder.eval()
        
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
        text_inputs = self.text_processor(text=text_queries, return_tensors="pt", padding=True).to(self.device)
        text_features = self.text_encoder.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def postprocess(self, outputs: Any, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        return {'detections': []}
    
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
                if detection.shape[0] < 5:
                    continue
                    
                box = detection[:4].cpu().numpy()
                obj_score = detection[4].cpu().numpy()
                
                if obj_score < 0.1:
                    continue
                
                x1, y1, x2, y2 = box
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.shape[1], int(x2))
                y2 = min(image.shape[0], int(y2))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                region = image[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                
                try:
                    region_inputs = self.text_processor(images=region, return_tensors="pt").to(self.device)
                    region_features = self.text_encoder.get_image_features(**region_inputs)
                    region_features = region_features / region_features.norm(dim=-1, keepdim=True)
                    
                    similarities = (region_features @ text_features.T).squeeze(0)
                    max_sim_idx = similarities.argmax().item()
                    max_sim = similarities[max_sim_idx].item()
                    
                    final_score = max_sim * obj_score
                    
                    if final_score > confidence_threshold:
                        results.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'score': float(final_score),
                            'class_id': max_sim_idx,
                            'class_name': text_queries[max_sim_idx]
                        })
                except Exception:
                    continue
        
        return {'detections': results}


class LEAFYOLO(SimpleYOLOBasedOVDetector):
    def __init__(self, device: str = "cuda"):
        super().__init__("LEAF-YOLO", device)


class SMDYOLO(SimpleYOLOBasedOVDetector):
    def __init__(self, device: str = "cuda"):
        super().__init__("SMD-YOLO", device)


class SimpleDETRBasedOVDetector(BaseOVDetector):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        self.text_encoder = None
        self.tokenizer = None
        
    def load_model(self, checkpoint_path: str = None):
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        
        model_name = "facebook/detr-resnet-50"
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        self.model.eval()
        self.text_encoder.eval()
        
    def preprocess(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        inputs = self.processor(images=image, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def get_text_features(self, text_queries: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(text_queries, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            text_features = outputs.last_hidden_state.mean(dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def postprocess(self, outputs: Any, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        return {'detections': []}
    
    def detect(self, image: np.ndarray, text_queries: List[str], confidence_threshold: float = 0.3) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.preprocess(image)
        text_features = self.get_text_features(text_queries)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]
        
        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        for box, score in zip(boxes, scores):
            if score < 0.1:
                continue
                
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.shape[1], int(x2))
            y2 = min(image.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            region_features = outputs.last_hidden_state.mean(dim=1)
            region_features = region_features / region_features.norm(dim=-1, keepdim=True)
            
            similarities = (region_features @ text_features.T).squeeze(0)
            max_sim_idx = similarities.argmax().item()
            max_sim = similarities[max_sim_idx].item()
            
            final_score = float(max_sim * score)
            
            if final_score > confidence_threshold:
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': final_score,
                    'class_id': max_sim_idx,
                    'class_name': text_queries[max_sim_idx]
                })
        
        return {'detections': detections}


class OVLWDETR(SimpleDETRBasedOVDetector):
    def __init__(self, device: str = "cuda"):
        super().__init__("OVLW-DETR", device)


class LightMDETR(SimpleDETRBasedOVDetector):
    def __init__(self, device: str = "cuda"):
        super().__init__("LightMDETR", device)


class DetCLIPv2(BaseOVDetector):
    def __init__(self, device: str = "cuda"):
        super().__init__("DetCLIPv2", device)
        self.clip_model = None
        self.clip_processor = None
        
    def load_model(self, checkpoint_path: str = None):
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        from ultralytics.nn.tasks import DetectionModel
        self.model = DetectionModel('yolov8n.yaml', ch=3, nc=80).to(self.device)
        
        self.model.eval()
        self.clip_model.eval()
        
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
    
    def postprocess(self, outputs: Any, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        return {'detections': []}
    
    def detect(self, image: np.ndarray, text_queries: List[str], confidence_threshold: float = 0.3) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        preprocessed_image = self.preprocess(image)
        
        text_inputs = self.clip_processor(text=text_queries, return_tensors="pt", padding=True).to(self.device)
        text_features = self.clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        with torch.no_grad():
            outputs = self.model(preprocessed_image)
        
        results = []
        
        for output in outputs:
            if len(output.shape) == 3:
                output = output[0]
            
            for detection in output:
                if detection.shape[0] < 5:
                    continue
                    
                box = detection[:4].cpu().numpy()
                obj_score = detection[4].cpu().numpy()
                
                if obj_score < 0.1:
                    continue
                
                x1, y1, x2, y2 = box
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.shape[1], int(x2))
                y2 = min(image.shape[0], int(y2))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                region = image[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                
                try:
                    region_inputs = self.clip_processor(images=region, return_tensors="pt").to(self.device)
                    region_features = self.clip_model.get_image_features(**region_inputs)
                    region_features = region_features / region_features.norm(dim=-1, keepdim=True)
                    
                    similarities = (region_features @ text_features.T).squeeze(0)
                    max_sim_idx = similarities.argmax().item()
                    max_sim = similarities[max_sim_idx].item()
                    
                    final_score = max_sim * obj_score * 1.2
                    
                    if final_score > confidence_threshold:
                        results.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'score': float(final_score),
                            'class_id': max_sim_idx,
                            'class_name': text_queries[max_sim_idx]
                        })
                except Exception:
                    continue
        
        return {'detections': results}