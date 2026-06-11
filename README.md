# Hough Transformation-based Neural Network for Particle Tracking

This research project is an advanced system based on deep Convolutional Neural Networks (CNN), designed for the precise reconstruction of elementary particle trajectories (e.g., muons) under extreme noise conditions. The system analyzes two-dimensional parameter space accumulators generated via the Hough Transform from real experimental data collected by CERN detectors. 

The algorithm successfully handles severe interaction overlaps (Pile-up 200) by filtering out the background noise and localizing physical phenomena using a **U-Net** architecture paired with a highly rigorous hybrid loss function (**Focal Loss + Dice Loss**).

## Main Features

* **Scalable Data Pipeline:** An asynchronous, on-the-fly generator based on `tf.data.Dataset` that loads `.npz` files from disk and extracts random 216x216 patches, preventing RAM overflow.
* **U-Net Architecture with BatchNormalization:** A modified encoder-decoder architecture enhanced with batch normalization layers, preventing vanishing gradients and mode collapse.
* **Robustness to Class Imbalance:** The model handles images where >99% of pixels are background noise by utilizing an innovative loss function that combines single-pixel stability (Focal Loss) with geometric enforcement of Gaussian distributions (Dice Loss).
* **Ground Truth Generation:** True coordinates of physical events are projected onto the matrix as smooth, 2D Gaussian distributions ($\sigma=5.0$), enabling stable network convergence.

---

## Repository Structure

The project evolved from a virtual environment (Monte Carlo) to operating on hard experimental data from CERN.

### Virtual and Mathematical Phase
* `MC.py` – Monte Carlo environment simulating ideal collisions and generating detector noise.
* `hough.py` – Mathematical apparatus for transforming Cartesian particle tracks into continuous curves within the parameter accumulator.

### Data Engineering (CERN Data)
* `read_data.py` – Explorer for `.npz` files, visualizing massive accumulators (14x7000x216).
* `data_ingestion.py` / `data_test.py` – Logic for slicing specific areas, correcting Cartesian axis systems, and populating thermal labels.

### Architecture and Training (Core AI)
* `count_analysis.py` – **The core of the project.** Defines U-Net encoders/decoders, the classic Sliding Window algorithm, and custom loss functions (BCE, Focal, Dice).
* `pipeline.py` – Integrates the infinite `tf.data` pipeline with training history plotting and advanced metrics (AUC-ROC, MAE).
* `train.py` – The main script overseeing the training process and saving optimized weights (`ModelCheckpoint`).

### Inference and Evaluation
* `inference.py` – A script to test the frozen, trained model on completely unknown, random data patches from the repository. Automatically compares the original noise with the smoothed U-Net prediction and algorithmically detected peaks.

---

## Installation and Requirements

The project requires Python 3.10+ and a dedicated virtual environment.

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/Hough-transformation-based-Neural-Network.git
cd Hough-transformation-based-Neural-Network

# 2. Install dependencies
pip install numpy matplotlib tensorflow
