# Fan
# Convolves clean speech with a room impulse response to simulate reverb,
# makes the model more robust to echoey rooms.
#
# TODO implement RIRAugmentor class and apply_rir function
#
# RIRAugmentor takes rir_json, prob=0.3, target_sr=16000. Load all RIR wav
# files listed in rir_json at init, store in a list. If any has a different
# sample rate resample to target_sr. prob=0.3 means 30% of training samples
# get reverb applied.
#
# __call__ rolls a random number, if above prob return audio unchanged.
# Otherwise pick a random RIR, fftconvolve audio with rir mode='full', trim
# to original length, energy-normalize so output has same RMS as input cause
# reverb changes loudness and we don't want that.
#
# apply_rir(audio, rir) is a standalone version without the probability check,
# just convolve + trim + normalize, in case dataloader calls it directly.
import json
import soundfile as sf
import librosa
import random
import scipy
import numpy as np

def apply_rir(audio, rir):
    if rir.ndim > 1:
        rir = rir[:, 0] if rir.shape[1] >= 1 else rir.ravel()
    rir = np.asarray(rir, dtype=np.float64).ravel()
    rst = scipy.signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
    audio_energy = np.sqrt(np.sum(audio**2))
    rst_energy = np.sqrt(np.sum(rst**2))
    if rst_energy <= 0:
        return audio
    rst = rst * (audio_energy / rst_energy)
    return rst

class RIRAugmentor:
    def __init__(self, rir_json, prob=0.3, target_sr=16000):
        #对rir_json进行解析，得到rir_list，保存概率和目标采样率
        with open(rir_json, 'r') as f:
            self.rir_list = json.load(f)
        self.rirs = []
        for rir in self.rir_list:
            rir_path = rir
            rir_data, sr = sf.read(rir_path)
            if sr != target_sr:
                rir_data = librosa.resample(rir_data, sr, target_sr)
            self.rirs.append(rir_data)
        self.prob = prob
        self.target_sr = target_sr

    def __call__(self, audio):
        #随机生成一个概率，如果概率大于等于prob，则返回原始音频
        if random.random() >= self.prob:
            return audio
        rir = random.choice(self.rirs)
        return apply_rir(audio, rir)

   
