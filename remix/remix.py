import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
import sys
print(sys.path)
sys.path.append('/home/hsinhung/speech_enhance_phasen/')

from util.metrics import STOI, PESQ, SI_SDR


enhanced_dir = "/home/hsinhung/speech_enhance_phasen/output/phasen_vb_wiener_alpha/enhanced/"
clean_dir = "/home/koredata/hsinhung/speech/vb_demand/clean_testset_wav/"
noisy_dir = "/home/koredata/hsinhung/speech/vb_demand/noisy_testset_wav/"

sample_rate = 16000

remix_pesq = []
remix_stoi = []
remix_sdr = []
noisy_sdr = []
enhanced_sdr = []

remix_ratio_noise = 0.8
remix_ratio_speech = 2
cnt = 0
#for path in enhanced_dir:
for (dirpath, dirnames, filenames) in os.walk(enhanced_dir):
    print(dirpath)
    for f in filenames:
        if not f.endswith((".WAV", ".wav")):
            continue
        #print(f)
        enhanced_path = enhanced_dir + f
        noisy_path = noisy_dir + f[:-10] + '.wav'
        clean_path = clean_dir + f[:-10] + '.wav'
        enhanced, _ = librosa.load(enhanced_path, sr=sample_rate)
        noisy, _ = librosa.load(noisy_path, sr=sample_rate)
        clean, _ = librosa.load(clean_path, sr=sample_rate)
        noise = noisy - enhanced

        remix = noisy + remix_ratio_speech * enhanced - remix_ratio_noise * noise
        remix = remix / np.abs(remix).max()

        #noisy_sdr_ = SI_SDR(clean, noisy)
        #enhanced_sdr_ = SI_SDR(clean, enhanced)
        remix_pesq_ = PESQ(clean, remix)
        remix_stoi_ = STOI(clean, remix)
        remix_sdr_ = SI_SDR(clean, remix)
        remix_pesq.append(remix_pesq_)
        remix_stoi.append(remix_stoi_)
        remix_sdr.append(remix_sdr_)
        #noisy_sdr.append(noisy_sdr_)
        #enhanced_sdr.append(enhanced_sdr_)

        print(f,':' )
        print('PESQ enhanced: ', PESQ(clean, enhanced), 'PESQ remix: ', remix_pesq_)
        print('STOI enhanced: ', STOI(clean, enhanced), 'STOI remix: ', remix_stoi_)
        print('SI_SDR enhanced: ', SI_SDR(clean, enhanced), 'SI_SDR remix: ', remix_sdr_)
        cnt += 1
        #if cnt > 10:
        #    break


print('remix PESQ: ', '{:.4f}'.format(np.mean(remix_pesq)), '±', '{:.4f}'.format(np.std(remix_pesq)))
print('remix STOI: ', '{:.4f}'.format(np.mean(remix_stoi)), '±', '{:.4f}'.format(np.std(remix_stoi)))
print('remix SI_SDR: ', '{:.4f}'.format(np.mean(remix_sdr)), '±', '{:.4f}'.format(np.std(remix_sdr)))

#print('NOISY SI-SDR: ', '{:.4f}'.format(np.mean(noisy_sdr)), '±', '{:.4f}'.format(np.std(noisy_sdr)))
#print('ENHANCED SI-SDR: ', '{:.4f}'.format(np.mean(enhanced_sdr)), '±', '{:.4f}'.format(np.std(enhanced_sdr)))
'''
NOISY PESQ: 1.9783 ± 0.7552
NOISY STOI: 0.9211 ± 0.0709
NOISY SI-SDR:  8.4465 ± 5.6149
ENHANCED PESQ: 2.8211 ± 0.6341
ENHANCED STOI: 0.9427 ± 0.0566
ENHANCED SI-SDR:  18.7953 ± 3.8375
'''
'''
remix_ratio_noise = 0.8
remix_ratio_speech = 0.8
remix PESQ:  2.7826 ± 0.6045
remix STOI:  0.9422 ± 0.0560
remix SI_SDR:  18.4122 ± 3.8514

remix_ratio_noise = 0.8
remix_ratio_speech = 1
remix PESQ:  2.7902 ± 0.6026
remix STOI:  0.9423 ± 0.0559
remix SI_SDR:  18.4706 ± 3.8459

remix_ratio_noise = 0.8
remix_ratio_speech = 1.2
remix PESQ:  2.7953 ± 0.6025
remix STOI:  0.9424 ± 0.0559
remix SI_SDR:  18.5153 ± 3.8425

remix_ratio_noise = 0.8
remix_ratio_speech = 2
remix PESQ:  2.8080 ± 0.6047
remix STOI:  0.9426 ± 0.0558
remix SI_SDR:  18.6208 ± 3.8373

wiener 1.1
remix_ratio_noise = 0.8
remix_ratio_speech = 2
remix PESQ:  2.8242 ± 0.6014
remix STOI:  0.9423 ± 0.0561
remix SI_SDR:  18.7350 ± 3.8360
'''