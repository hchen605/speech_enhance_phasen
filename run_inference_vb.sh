#!/bin/bash



CUDA_VISIBLE_DEVICES=1 python inference.py -mc /home/hsinhung/SE-D2_deliverables/Experiments/phasen_vb/config.json \
    -dc config/inference/test_vb.json \
    -cp /home/hsinhung/SE-D2_deliverables/Experiments/phasen_vb/checkpoints/latest_model.tar \
    -dist phasen_vb