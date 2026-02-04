
# Robust Road Scene Segmentation Under Adverse Conditions

This project investigates semantic segmentation for autonomous driving scenes under challenging visual conditions such as nighttime, rain, fog, and snow.

We train and evaluate multiple*deep learning segmentation architectures to study the accuracy–latency tradeoffs and failure modes that arise in safety-critical perception systems.

The goal is to understand **how different model designs behave under adverse conditions**, and what architectural choices matter most for real-world deployment.

---

## Table of Contents

- Paper
- Features  
- Dataset & Preprocessing  
- Installation  
- Usage  
- Evaluation & Metrics  
- Technologies Used  
- Results  

---

## Project Paper (Click View Full)

[![View the paper](report-preview.png)](report.pdf)

---

## Features

- Semantic segmentation of road scenes using multiple architectures  
- Comparative analysis of:
  - High-accuracy vs lightweight models  
  - Performance under adverse weather and low-light conditions  
- Condition-aware evaluation (night vs adverse weather vs overall)  
- Quantitative metrics:
  - Mean Intersection-over-Union (mIoU)
  - Pixel Accuracy
  - Inference latency (FPS)
- Modular training and evaluation pipeline

---

## Dataset & Preprocessing

### Dataset
- BDD100K Semantic Segmentation
  - RGB driving images with pixel-level labels
  - Metadata including time of day, weather, and scene type

### Data Structure
Each sample consists of:
- Image: RGB `.jpg` (C × H × W)
- Mask: Semantic label `.png` (H × W)

### Preprocessing Pipeline
Applied consistently to all ~7,000 labeled frames:

- Resize images and masks to a fixed resolution (e.g., 512×512)
- Normalize pixel values to stabilize training
- Convert masks to integer class indices
- Apply the same transforms to both image and mask to preserve alignment

A PyTorch `Dataset` + `DataLoader` handles:
- Randomized shuffling each epoch
- Mini-batching
- Streaming data efficiently to GPU / Apple MPS

---

## Features

- Pixel-level semantic segmentation of driving scenes
- Comparison of high-accuracy and lightweight models
- Evaluation under adverse conditions:
  - Nighttime
  - Rain, snow, fog, overcast
- Quantitative metrics:
  - Mean Intersection-over-Union (mIoU)
  - Pixel Accuracy
  - Inference latency / FPS
- Modular training and evaluation pipeline

## ⚙️ Installation

### Prerequisites

Ensure the following are installed:

- Python ≥ 3.9
- PyTorch
- torchvision
- NumPy
- OpenCV
- matplotlib

### Clone the Repository

$ git clone https://github.com/your-username/road-segmentation-adverse-weather.git  
$ cd road-segmentation-adverse-weather

### Install Dependencies

$ pip install -r requirements.txt

---

## Usage

### Train a Model

$ python -m src.training.train deeplab

Available models:
- deeplab
- fastscnn
- mobilenet
- baselinecnn

### Evaluate a Model

$ python -m src.evaluation.eval deeplab

### Condition-Based Evaluation

$ python -m src.evaluation.eval_conditions \
  data/bdd100k_labels_images_train.json \
  data/bdd100k_labels_images_val.json

---

## Evaluation & Metrics

### Metrics

- Mean Intersection-over-Union (mIoU)
  Measures overlap between predicted and ground-truth segments

- Pixel Accuracy
  Fraction of correctly labeled pixels

- Inference Latency / FPS
  Measures real-time deployment feasibility

### Condition-Based Analysis

Performance is reported for:
- All validation frames
- Nighttime scenes
- Adverse weather scenes

This mirrors evaluation practices used in autonomous vehicle perception teams.

---

## Technologies Used

- Python
- PyTorch
- torchvision
- NumPy
- OpenCV
- matplotlib
- Apple Metal (MPS backend)

---

##  Models

### Baseline CNN
- Simple convolutional encoder
- Serves as a reference for naive segmentation approaches
- Demonstrates limitations of basic CNNs for spatially dense prediction

### DeepLabV3
- High-accuracy architecture
- Captures global context while preserving spatial detail
- Higher computational cost

### MobileNetV3
- Efficient architecture for embedded systems
- Designed for low-power, real-time inference

---

## Results

Final results include:
- Tables comparing mIoU and pixel accuracy
- Accuracy vs latency plots
- Condition-specific performance breakdowns
- Qualitative analysis of failure cases

The study highlights:
- Clear accuracy–speed tradeoffs
- Performance degradation under low-light and adverse weather
- Architectural strengths and weaknesses

---

## Contributing

Worked with Jonathan Wang at Grinnell College


