# speech_enhance_phasen

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --configuration config/train_RATS/CRN_vb.json
```
## Inference

```bash
source run_inference_vb.sh
```

NOISY PESQ: 1.9783 ± 0.7552
NOISY STOI: 0.9211 ± 0.0709
ENHANCED PESQ: 2.8100 ± 0.6717
ENHANCED STOI: 0.9419 ± 0.0542