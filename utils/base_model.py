import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class BaseOVDetector(ABC):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        
    @abstractmethod
    def load_model(self, checkpoint_path: str = None):
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        pass
    
    @abstractmethod
    def postprocess(self, outputs: Any, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray, text_queries: List[str], confidence_threshold: float = 0.3) -> Dict[str, Any]:
        pass
    
    def to_onnx(self, output_path: str, dummy_input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        dummy_input = torch.randn(dummy_input_shape).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )