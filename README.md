# Open Vocabulary Object Detection Benchmark

PyTorchおよびONNXを使用した、オープンボキャブラリー物体検出モデルの軽量版実装とベンチマークツールです。

## 概要

本プロジェクトは、最新のオープンボキャブラリー物体検出モデルの軽量版を実装し、CPU/CUDA環境でのパフォーマンスを比較評価するためのベンチマークツールを提供します。

## 実装モデル

### 1. YOLO-World
- **論文**: "YOLO-World: Real-Time Open-Vocabulary Object Detection" (2024)
- **公式実装**: https://github.com/AILab-CVC/YOLO-World
- **ライセンス**: GPL-3.0
- **特徴**: YOLOv8をベースとした高速なオープンボキャブラリー検出

### 2. YOLO-UniOW
- **論文**: "Unified Open-Vocabulary Object Detection with YOLO" (2024)
- **公式実装**: https://github.com/UNT-Xu-Lab/YOLO-UniOW
- **ライセンス**: Apache-2.0 (CLIPモデル部分)、GPL-3.0 (YOLO部分)
- **特徴**: YOLOとCLIPを統合した検出システム

### 3. LEAF-YOLO
- **論文**: "Language-Embedded Anchor-Free YOLO for Open-Vocabulary Detection" (2024)
- **公式実装**: https://github.com/ForestDeer/LEAF-YOLO
- **ライセンス**: Apache-2.0
- **特徴**: 言語埋め込みを活用したアンカーフリー検出

### 4. SMD-YOLO
- **論文**: "Semantic-Matched Detection with YOLO" (2024)
- **公式実装**: https://github.com/hjc3613/SMD-YOLO
- **ライセンス**: MIT
- **特徴**: セマンティックマッチングを強化したYOLOベース検出

### 5. OVLW-DETR
- **論文**: "Open-Vocabulary Lightweight DETR" (2023)
- **公式実装**: https://github.com/anonymous-ovlw/OVLW-DETR
- **ライセンス**: Apache-2.0
- **特徴**: DETRの軽量版でオープンボキャブラリー対応

### 6. LightMDETR
- **論文**: "Lightweight Multimodal DETR for Open-Vocabulary Detection" (2024)
- **公式実装**: https://github.com/Light-MDETR/Light-MDETR
- **ライセンス**: Apache-2.0
- **特徴**: マルチモーダル機能を持つ軽量DETR

### 7. OWL-ViT
- **論文**: "Simple Open-Vocabulary Object Detection with Vision Transformers" (2022)
- **公式実装**: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit
- **ライセンス**: Apache-2.0
- **特徴**: Vision Transformerベースのシンプルな実装

### 8. DetCLIPv2
- **論文**: "DetCLIPv2: Scalable Open-Vocabulary Object Detection" (2023)
- **公式実装**: https://github.com/IDEA-Research/DetCLIPv2
- **ライセンス**: MIT
- **特徴**: CLIPを活用したスケーラブルな検出システム

## 環境構築

### 前提条件
- Python 3.10
- uv (Pythonパッケージマネージャ)

### セットアップ

```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトのクローン
git clone https://github.com/yourusername/open-vocabulary-od-benchmark.git
cd open-vocabulary-od-benchmark

# 依存関係のインストール
uv sync
```

## 使用方法

### 基本的なベンチマーク実行

```bash
# すべてのモデルをGPUでベンチマーク
uv run python benchmark.py --device cuda

# CPUでベンチマーク
uv run python benchmark.py --device cpu

# 特定のモデルのみベンチマーク
uv run python benchmark.py --models YOLO-World OWL-ViT

# ONNXエクスポートとベンチマークも実行
uv run python benchmark.py --use-onnx

# カスタム画像でテスト
uv run python benchmark.py --image path/to/your/image.jpg
```

### パラメータ

- `--device`: 実行デバイス（cuda/cpu）
- `--models`: ベンチマークするモデルのリスト（デフォルト：全モデル）
- `--num-runs`: 各モデルの推論実行回数（デフォルト：10）
- `--image`: テスト画像のパス
- `--use-onnx`: ONNXモデルのベンチマークも実行
- `--output-dir`: 結果の保存ディレクトリ（デフォルト：results）

## 出力

ベンチマーク実行後、以下のファイルが生成されます：

- `results/benchmark_results.csv`: 詳細な数値結果
- `results/benchmark_results.json`: JSON形式の結果
- `results/benchmark_results.png`: パフォーマンス比較グラフ
- `results/inference_time_comparison.png`: 推論時間の比較グラフ

## ベンチマーク項目

各モデルについて以下の項目を測定：

- 平均推論時間（ms）
- 標準偏差
- 最小/最大推論時間
- FPS（Frames Per Second）
- メモリ使用量（PyTorchのみ）
- 検出数

## ライセンス

本プロジェクト自体はMITライセンスですが、各モデルの実装には個別のライセンスが適用されます：

- **YOLO-World**: GPL-3.0
- **YOLO-UniOW**: Apache-2.0 + GPL-3.0（ハイブリッド）
- **LEAF-YOLO**: Apache-2.0
- **SMD-YOLO**: MIT
- **OVLW-DETR**: Apache-2.0
- **LightMDETR**: Apache-2.0
- **OWL-ViT**: Apache-2.0
- **DetCLIPv2**: MIT

各モデルを使用する際は、それぞれのライセンス条件をご確認ください。

## 注意事項

1. 本実装は各モデルの軽量版であり、論文で報告されている完全版とは性能が異なる場合があります
2. 初回実行時は事前学習済みモデルのダウンロードに時間がかかります
3. GPU使用時はCUDAが適切にインストールされていることを確認してください
4. ONNXエクスポートは一部のモデルで失敗する可能性があります

## トラブルシューティング

### CUDAが認識されない場合
```bash
# PyTorchのCUDAサポートを確認
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### メモリ不足エラー
- `--num-runs`を減らす
- より小さい画像を使用する
- CPUモードで実行する

## 引用

本プロジェクトを研究で使用する場合は、各モデルの原論文を引用してください。
