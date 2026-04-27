# VSL-GCN

**Official implementation of: Enhancing Vietnamese Sign Language Recognition via Cross-Modal Transfer Learning and Multi-Stream Ensemble based on CTR-GCN.**

## Introduction

Vietnamese Sign Language (VSL) recognition faces significant hurdles due to the lack of large-scale labeled datasets and high inter-class similarity. Traditional methods often fail to capture the subtle nuances of VSL gestures.

This project introduces **VSL-GCN**, a comprehensive framework that integrates a robust preprocessing pipeline with a novel training strategy. Our core contributions include:

**1.  Greedy Shape-Matching Transfer:** A novel weight loading mechanism that bypasses variable naming mismatches and enables partial channel loading, allowing the model to leverage >90% of pre-trained knowledge even when input channels differ (9-channel fusion vs. 3-channel original).

**2.  "Clone & Evolve" Strategy (Cross-Modal Initialization):** Instead of training Bone and Velocity streams from scratch or incompatible weights, we propose a cross-modal transfer method. We "clone" the weights from the converged Joint stream to initialize Bone and Velocity streams. This addresses the "cold-start" problem and significantly boosts the accuracy of secondary streams (e.g., Bone stream improvement from 41% to ~76%).

**3.  Robust Multi-Stream Ensemble with TTA:** A fusion framework that combines Joint, Bone, and Velocity streams using optimized weights, further enhanced by Test-Time Augmentation (TTA) to achieve state-of-the-art performance on our custom VSL dataset.

## Performance

Experiments were conducted on our self-collected VSL Dataset containing 200 classes.

| Method | Stream | Strategy | Top-1 Acc (%) | Top-5 Acc (%) | Checkpoint |
| :--- | :---: | :--- | :---: | :---: | :---: |
| Baseline (NTU Weights) | Joint | Direct Transfer | 73.36 | 85.12 | [link](#) |
| Baseline (NTU Weights) | Bone | Direct Transfer | 41.67 | 62.40 | - |
| **Proposed (Ours)** | **Joint** | **Shape-Matching** | **76.36** | **90.15** | **[link](#)** |
| **Proposed (Ours)** | **Bone** | **Clone & Evolve** | **76.61** | **90.88** | **[link](#)** |
| **Proposed (Ours)** | **Velocity** | **Clone & Evolve** | **75.73** | **89.50** | **[link](#)** |
| Ensemble (3-Stream) | Fusion | Weighted Sum | 77.37 | 91.80 | - |
| **Ensemble + TTA** | **Fusion** | **TTA x5** | **77.50** | **92.04** | **Best** |

> **Note:** The "Clone & Evolve" strategy improved the Bone stream accuracy by **+35%** compared to the baseline transfer method.

## Data Preparation
The VSL-GCN framework requires a specific data format (9-channel input: Joint + Velocity + Bone).
We provide a comprehensive **4-Stage Pipeline** to process raw videos into model-ready features.

**Please follow the detailed instructions in [DATA.md](DATA.md).**

**Summary of the pipeline:**

 **1.  Stage 1 (YOLOv8):** Detect humans and crop ROI to remove background noise.

 **2.  Stage 2 (MediaPipe):** Extract 46 holistic keypoints (Hands + Upper Body).

 **3.  Stage 3 (Fusion):** Normalize, Interpolate, Augment (x6), and Fuse data into 9 channels.

### 1. Feature Extraction
We use 9-channel input fusion:
* **Channels 0-2:** Joint coordinates ```(x, y, z)```.
* **Channels 3-5:** Velocity information.
* **Channels 6-8:** Bone vectors.

### 2. Directory Structure
Please ensure your data is organized as follows (or follow instructions in `DATA.md`):

```text
data/
├── MultiVSL200_videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...             # Other video files
└──  Multi-VSL200/
│    └── Multi-VSL200/
│        ├── train_labels.csv
│        ├── val_labels.csv
│        └── test_labels.csv
├── detection_yolo/      # Stage 1 scripts
├── mediapipe_kpt/       # Stage 2 scripts 
├── fusion/              # Stage 3 scripts
└── ...
```
## Usage
### Package
Install the required dependencies:

```pip install -r requirements.txt```

### Training
**Step 1: Train Joint Stream (The Teacher)** Train the Joint stream using Greedy Shape-Matching transfer from NTU weights.
```text
python -m trainer.trainer --config configs/transfer_joint.yaml
```

**Step 2: Train Bone Stream (Clone)** Use the best Joint model to initialize and train the Bone stream.

*Note: We set TARGET_STREAM=BONE to force weight copying from Joint channels (0-2) to Bone channels (6-8).*
```text
# Point 'pretrained_path' in config to: ./results/transfer_joint/best_model.pth
TARGET_STREAM=BONE python -m trainer.trainer --config configs/transfer_bone.yaml
```

**Step 3: Train Velocity Stream (Clone)** Use the best Joint model to initialize and train the Velocity stream.
```text
# Point 'pretrained_path' in config to: ./results/transfer_joint/best_model.pth
TARGET_STREAM=VELOCITY python -m trainer.trainer --config configs/transfer_vel.yaml
```
### Testing (Single Stream Evaluation)
Use this step to evaluate the performance of individual streams separately on the Test set.

**Test Joint Stream:**
```text
python -m trainer.trainer --config configs/transfer_joint.yaml --phase test --weights ./results/transfer_joint/best_model.pth
```
**Test Bone Stream:**
```text
python -m trainer.trainer --config configs/transfer_bone.yaml --phase test --weights ./results/transfer_bone/best_model.pth
```
**Test Velocity Stream:**
```text
python -m trainer.trainer --config configs/transfer_vel.yaml --phase test --weights ./results/transfer_vel/best_model.pth
```
### Ensemble (Multi-Stream Fusion)
To achieve the state-of-the-art result (77.50%), run the Ensemble script. This script fuses the 3 streams and applies Test-Time Augmentation (TTA).
```text
python ensemble_tta.py \
  --config1 configs/transfer_joint.yaml \
  --weight1 ./results/checkpoints/transfer_joint/best_model.pth \
  --config2 configs/transfer_bone.yaml \
  --weight2 ./results/checkpoints/transfer_bone/best_model.pth \
  --config3 configs/transfer_vel.yaml \
  --weight3 ./results/checkpoints/transfer_vel/best_model.pth \
  --tta_times 5
```
***Output**: The script will print the best fusion accuracy and generate a Confusion Matrix at ./results/confusion_matrix_final.png.*
## Acknowledgement
This project is built by iBME Lab at School of Electrical & Electronic Engineering, Hanoi University of Science and Technology, Vietnam. It is based on the CTR-GCN framework. We thank the authors for their open-source contribution.

python ensemble_tta.py --config1 configs/transfer_joint.yaml --weight1 ./results1/noJDMAreal/transfer_joint/best_model.pth --config2 configs/transfer_bone.yaml --weight2 ./results1/noJDMA/transfer_bone/best_model.pth --config3 configs/transfer_vel.yaml --weight3 ./results1/noJDMA/transfer_vel/best_model.pth --tta_times 5