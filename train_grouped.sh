#!/bin/bash

# Generate a unique ID for the wandb run group
WANDB_RUN_ID=$(python -c "import uuid; print(uuid.uuid4())")

# 1. Train Joint model
echo "--- Training Joint Model ---"
python trainer/trainer.py --config configs/transfer_joint.yaml --wandb.id $WANDB_RUN_ID --wandb.group "transfer-learning" --wandb.enable True

# Check if the first command was successful
if [ $? -ne 0 ]; then
    echo "Joint model training failed. Exiting."
    exit 1
fi

echo "--- Joint Model Training Finished ---"

# The best model path from the joint training
BEST_MODEL_PATH="./results/transfer_joint/best_model.pth"

# 2. Train Bone model
echo "--- Training Bone Model ---"
# Update the pretrained_path in the bone config before running
# This is a temporary modification for the run
TARGET_STREAM=BONE python -m trainer.trainer --config configs/transfer_bone.yaml \
    --model_args.pretrained_path $BEST_MODEL_PATH \
    --wandb.id $WANDB_RUN_ID --wandb.group "transfer-learning" --wandb.enable True --wandb.resume allow

if [ $? -ne 0 ]; then
    echo "Bone model training failed. Exiting."
    exit 1
fi

echo "--- Bone Model Training Finished ---"


# 3. Train Velocity model
echo "--- Training Velocity Model ---"
# Update the pretrained_path in the velocity config before running
TARGET_STREAM=VELOCITY python -m trainer.trainer --config configs/transfer_vel.yaml \
    --model_args.pretrained_path $BEST_MODEL_PATH \
    --wandb.id $WANDB_RUN_ID --wandb.group "transfer-learning" --wandb.enable True --wandb.resume allow

if [ $? -ne 0 ]; then
    echo "Velocity model training failed. Exiting."
    exit 1
fi

echo "--- Velocity Model Training Finished ---"

echo "--- All training finished successfully! ---"
