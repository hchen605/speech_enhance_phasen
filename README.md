# speech_enhance_phasen

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --configuration config/train_RATS/phasen_vb.json
```
## Inference

```bash
source run_inference_vb.sh
```

# Current test result

## VoiceBank+DEMAND dataset
NOISY PESQ: 1.9783 ± 0.7552
NOISY STOI: 0.9211 ± 0.0709
NOISY SI-SDR:  8.4465 ± 5.6149

## PHASEN baseline (40 epochs)
PESQ = 2.8211 ± 0.6341
STOI = 0.9427 ± 0.0566
SI-SDR = 18.7953 ± 3.8375

## PHASEN + vanilla Wiener Filter (alpha = 1.1)
PESQ = 2.8229 ± 0.6373
STOI = 0.9425 ± 0.0568

## PHASEN + vanilla Wiener Filter + vanilla Remix 
PESQ = 2.8242 ± 0.6014
STOI = 0.9423 ± 0.0561
SI_SDR = 18.7350 ± 3.8360
