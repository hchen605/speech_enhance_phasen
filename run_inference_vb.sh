#!/bin/bash



CUDA_VISIBLE_DEVICES=0 python inference.py -mc /home/hsinhung/SE-D2_deliverables/Experiments/CRN_vb/config.json \
    -dc config/inference/test_vb.json \
    -cp /home/hsinhung/SE-D2_deliverables/Experiments/CRN_vb/checkpoints/latest_model.tar \
    -dist CRN_vb