#!/usr/bin/env python3
import os
import time
import json
import argparse
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any

from models.yolo_world import YOLOWorld
from models.yolo_uniow import YOLOUniOW
from models.owl_vit import OWLViT
from models.simple_od_models import LEAFYOLO, SMDYOLO, OVLWDETR, LightMDETR, DetCLIPv2


MODEL_CLASSES = {
    'YOLO-World': YOLOWorld,
    'YOLO-UniOW': YOLOUniOW,
    'LEAF-YOLO': LEAFYOLO,
    'SMD-YOLO': SMDYOLO,
    'OVLW-DETR': OVLWDETR,
    'LightMDETR': LightMDETR,
    'OWL-ViT': OWLViT,
    'DetCLIPv2': DetCLIPv2
}


class Benchmark:
    def __init__(self, device: str = "cuda", use_onnx: bool = False):
        self.device = device
        self.use_onnx = use_onnx
        self.results = {}
        
    def load_test_image(self, image_path: str) -> np.ndarray:
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            image = Image.fromarray(image)
        return np.array(image)
    
    def benchmark_pytorch_model(self, model_name: str, model, image: np.ndarray, text_queries: List[str], num_runs: int = 10) -> Dict[str, Any]:
        print(f"\nBenchmarking {model_name} (PyTorch)...")
        
        warmup_runs = 3
        for _ in range(warmup_runs):
            _ = model.detect(image, text_queries)
        
        inference_times = []
        memory_usage = []
        
        for _ in tqdm(range(num_runs), desc=f"{model_name} PyTorch"):
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            start_time = time.time()
            results = model.detect(image, text_queries)
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            if self.device == "cuda":
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)
                torch.cuda.reset_peak_memory_stats()
        
        return {
            'model_name': model_name,
            'backend': 'PyTorch',
            'device': self.device,
            'mean_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'fps': 1000 / np.mean(inference_times),
            'mean_memory_mb': np.mean(memory_usage) if memory_usage else 0,
            'num_detections': len(results.get('detections', []))
        }
    
    def benchmark_onnx_model(self, model_name: str, onnx_path: str, image: np.ndarray, text_queries: List[str], num_runs: int = 10) -> Dict[str, Any]:
        print(f"\nBenchmarking {model_name} (ONNX)...")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        input_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        input_numpy = input_tensor.numpy()
        
        warmup_runs = 3
        for _ in range(warmup_runs):
            _ = session.run(None, {session.get_inputs()[0].name: input_numpy})
        
        inference_times = []
        
        for _ in tqdm(range(num_runs), desc=f"{model_name} ONNX"):
            start_time = time.time()
            outputs = session.run(None, {session.get_inputs()[0].name: input_numpy})
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
        
        return {
            'model_name': model_name,
            'backend': 'ONNX',
            'device': 'CUDA' if 'CUDAExecutionProvider' in providers else 'CPU',
            'mean_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'fps': 1000 / np.mean(inference_times),
            'mean_memory_mb': 0,
            'num_detections': 0
        }
    
    def run_benchmark(self, models_to_test: List[str] = None, image_path: str = None, num_runs: int = 10):
        if models_to_test is None:
            models_to_test = list(MODEL_CLASSES.keys())
        
        test_image = self.load_test_image(image_path or "test_image.jpg")
        text_queries = ["person", "car", "dog", "chair", "bottle"]
        
        all_results = []
        
        for model_name in models_to_test:
            if model_name not in MODEL_CLASSES:
                print(f"Unknown model: {model_name}")
                continue
            
            try:
                model_class = MODEL_CLASSES[model_name]
                model = model_class(device=self.device)
                model.load_model()
                
                pytorch_results = self.benchmark_pytorch_model(
                    model_name, model, test_image, text_queries, num_runs
                )
                all_results.append(pytorch_results)
                
                if self.use_onnx:
                    onnx_path = f"models/{model_name.lower().replace('-', '_')}.onnx"
                    try:
                        model.to_onnx(onnx_path)
                        onnx_results = self.benchmark_onnx_model(
                            model_name, onnx_path, test_image, text_queries, num_runs
                        )
                        all_results.append(onnx_results)
                    except Exception as e:
                        print(f"ONNX export/benchmark failed for {model_name}: {e}")
                
            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
                continue
        
        self.results = pd.DataFrame(all_results)
        return self.results
    
    def save_results(self, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        
        self.results.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
        
        with open(os.path.join(output_dir, "benchmark_results.json"), 'w') as f:
            json.dump(self.results.to_dict('records'), f, indent=2)
        
        self.plot_results(output_dir)
    
    def plot_results(self, output_dir: str):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        df = self.results
        
        ax = axes[0, 0]
        df_pivot = df.pivot(index='model_name', columns='backend', values='mean_inference_time_ms')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Mean Inference Time by Model and Backend')
        ax.set_ylabel('Time (ms)')
        ax.set_xlabel('Model')
        ax.legend(title='Backend')
        ax.tick_params(axis='x', rotation=45)
        
        ax = axes[0, 1]
        df_pivot = df.pivot(index='model_name', columns='backend', values='fps')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('FPS by Model and Backend')
        ax.set_ylabel('FPS')
        ax.set_xlabel('Model')
        ax.legend(title='Backend')
        ax.tick_params(axis='x', rotation=45)
        
        ax = axes[1, 0]
        pytorch_df = df[df['backend'] == 'PyTorch']
        if not pytorch_df.empty:
            pytorch_df.plot(x='model_name', y='mean_memory_mb', kind='bar', ax=ax, legend=False)
            ax.set_title('Memory Usage (PyTorch only)')
            ax.set_ylabel('Memory (MB)')
            ax.set_xlabel('Model')
            ax.tick_params(axis='x', rotation=45)
        
        ax = axes[1, 1]
        summary_data = []
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            pytorch_row = model_df[model_df['backend'] == 'PyTorch']
            if not pytorch_row.empty:
                summary_data.append({
                    'Model': model,
                    'PyTorch FPS': pytorch_row['fps'].values[0],
                    'Memory (MB)': pytorch_row['mean_memory_mb'].values[0]
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('PyTorch FPS', ascending=False)
            
            cell_text = []
            for _, row in summary_df.iterrows():
                cell_text.append([row['Model'], f"{row['PyTorch FPS']:.1f}", f"{row['Memory (MB)']:.1f}"])
            
            table = ax.table(cellText=cell_text,
                           colLabels=['Model', 'FPS', 'Memory (MB)'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax.axis('off')
            ax.set_title('Performance Summary (PyTorch)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'benchmark_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        pytorch_df = df[df['backend'] == 'PyTorch'].sort_values('mean_inference_time_ms')
        if not pytorch_df.empty:
            plt.barh(pytorch_df['model_name'], pytorch_df['mean_inference_time_ms'])
            plt.xlabel('Inference Time (ms)')
            plt.ylabel('Model')
            plt.title('Model Inference Time Comparison (PyTorch)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Open Vocabulary Object Detection Benchmark')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run benchmark on')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to benchmark (default: all)')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of inference runs per model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image')
    parser.add_argument('--use-onnx', action='store_true',
                        help='Also benchmark ONNX models')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    benchmark = Benchmark(device=args.device, use_onnx=args.use_onnx)
    results = benchmark.run_benchmark(
        models_to_test=args.models,
        image_path=args.image,
        num_runs=args.num_runs
    )
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(results.to_string(index=False))
    
    benchmark.save_results(args.output_dir)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()