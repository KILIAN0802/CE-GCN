    # Data Preparation 

    This document describes the end-to-end pipeline to prepare data for **VSL-GCN**.
    The system processes raw videos through 4 main stages to generate **9-channel fused features** ready for training.

    ##  Directory Structure

    Before running the scripts, organize your data as follows:
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
    ## Multi-VSL
    The Multi-VSL dataset can be downloaded from [here](https://drive.google.com/drive/folders/1yUU1m2hy_CjaXDDoR_6i9Y3T1XL2pD4C).

    The lookup table (CSV file) mapping glosses to label IDs and their meanings is available [here](data/MultiVSL200).

    ## Stage 1: ROI Cropping (yolov8n-pose)
    **Goal**: Detect the signer and crop the upper-body region of interest (ROI) to remove background noise and improve MediaPipe accuracy
    * Input: Raw videos.
    * Model: yolov8n-pose.pt
    * Output: Resized (256x256), centered, cropped videos.
    **Run command**:
    ```text
    python detection_yolo/crop_video_list.py \
        --model_path detection_yolo/weights/yolov8n-pose.pt \
        --raw_dir   ./data/MultiVSL200_videos \
        --save_dir  data/cropped_videos \
        --train_csv data/Multi-VSL200/Multi-VSL200/train_labels.csv \
        --val_csv   data/Multi-VSL200/Multi-VSL200/val_labels.csv \
        --test_csv  data/Multi-VSL200/Multi-VSL200/test_labels.csv
    ```
    ## Stage 2: Skeleton Extraction (MediaPipe)
    **Goal**: Extract holistic keypoints from cropped videos.We extract a total of **46 keypoints**: 
    * 21 points: Left Hand6.
    * 21 points: Right Hand7.
    * 4 points: Upper Body (Shoulders, Neck, Hip).

    **Run command**:
    ```text
    python mediapipe_kpt/extract_46_keypoints.py \
        --video_dir ./data/cropped_videos \
        --out_dir /data/keypoints
    ```
    Output: ```.npy``` files with shape ```(T, 46, 3)``` (T: number of frames).
    ## Stage 3: Advanced Preprocessing & Fusion
    **Goal**: Handle noise, normalize poses, augment training data, and generate multi-stream features.

    **Key Techniques applied**:

    **1.** Pose Normalization: Align the central joint (Spine/Hip) to (0,0) and scale based on shoulder width.

    **2.** Missing Data Handling: Interpolate missing frames from MediaPipe.

    **3.** Noise Filtering: Apply Kalman Filter to smooth joint movements.

    **4.** Offline Augmentation (Train set only): Generate 6 variations per sample using Random Rotation, Scaling, and Gaussian Noise.

    **5 .** Early Fusion: Combine 3 information streams into a single tensor15:
    * Joint: Absolute coordinates ```(x, y, z)```.
    * Velocity: Motion speed over time.
    * Bone: Spatial vectors between connected joints.
    * Total: 9 Channels.
    
    **Run command:**
    ```text 
    python generate_fused_features.py 
    ```

    ## Final Output
    After Stage 3, the data is ready for training. 
    The output directory ```data/fused_features/``` will look like this:
    ```
    data/fused_features/
    ├── train_fused_features/   # ~25,000 files (Original + Augmented)
    │   ├── video_001.npy
    │   ├── ...
    │   └── train_label.csv
    ├── val_fused_features/     # Original validation set
    │   ├── ...
    │   └── val_label.csv
    └── test_fused_features/    # Original test set
        ├── ...
        └── test_label.csv
    ```