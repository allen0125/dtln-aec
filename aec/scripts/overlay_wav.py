import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve

# ----------------------------------------------------------------------------
# Echo synthesis for DTLN-aec training data.
#
# The microphone signal of an AEC scenario is  y(n) = s(n) + d(n) (+ noise),
# where s(n) is the near-end speech (the training target) and d(n) is the
# acoustic echo, i.e. the far-end / loopback signal after it travelled through
# the loudspeaker-room-microphone (LRM) path.
#
# Simply overlaying the raw loopback onto the mic (the previous behaviour) lets
# the network learn a trivial copy-subtraction and does not generalize to real
# echoes. Instead we synthesize a realistic echo from the loopback signal,
# following the DTLN-aec paper:
#   1. convolve the loopback with a (random or measured) room impulse response,
#   2. attenuate the reverberant tail by a random gain in [-25, 0] dB,
#   3. add a random bulk delay,
#   4. band-pass filter to mimic the loudspeaker response
#      (low cut 100-400 Hz, high cut 6000-7500 Hz),
#   5. mix near-end + echo at a random signal-to-echo ratio (SER).
#
# The loopback file used for training stays the *raw* far-end reference - only
# the mixed file contains the processed echo; the network has to estimate the
# echo path itself.
# ----------------------------------------------------------------------------

fs = 16000
mix_output_dir = "/home/allen/Data/ainur/aec/train_data_aligned/mixed"
# optional directory with measured room impulse responses (*.wav, mono).
# if empty / missing, a random exponentially decaying RIR is generated instead.
rir_dir = ""
# range of the signal-to-echo ratio in dB (near-end power vs. echo power)
ser_range_db = (-10.0, 10.0)


def _load_rirs(directory):
    rirs = []
    if directory and os.path.isdir(directory):
        for f in os.listdir(directory):
            if f.lower().endswith(".wav"):
                h, sr = sf.read(os.path.join(directory, f))
                if h.ndim > 1:
                    h = h[:, 0]
                if sr != fs:
                    import librosa

                    h = librosa.resample(h.astype("float32"), sr, fs)
                rirs.append(h.astype("float32"))
    return rirs


def _random_rir(length=int(0.3 * fs), rt60=0.2):
    # exponentially decaying gaussian noise as a synthetic RIR
    t = np.arange(length) / fs
    decay = np.exp(-6.908 * t / rt60)  # -60 dB after rt60 seconds
    h = np.random.randn(length).astype("float32") * decay
    h[0] = 1.0  # strong direct path
    return h


def _bandpass(x, low, high):
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, x).astype("float32")


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def synth_echo(lpb, rirs):
    # 1. pick / build an impulse response, normalize the direct path to 1
    if rirs:
        h = rirs[np.random.randint(len(rirs))].copy()
    else:
        h = _random_rir()
    h = h / (np.abs(h[0]) + 1e-12)
    # 2. attenuate the reverberant tail by a random gain in [-25, 0] dB
    gain_db = np.random.uniform(-25.0, 0.0)
    h[1:] = h[1:] * (10.0 ** (gain_db / 20.0))
    # convolve the far-end with the LRM path
    echo = fftconvolve(lpb, h)[: len(lpb)].astype("float32")
    # 3. random bulk delay (0-40 ms)
    delay = np.random.randint(0, int(0.04 * fs) + 1)
    if delay:
        echo = np.concatenate([np.zeros(delay, dtype="float32"), echo])[: len(lpb)]
    # 4. band-pass filter to mimic the loudspeaker response
    low = np.random.uniform(100.0, 400.0)
    high = np.random.uniform(6000.0, 7500.0)
    echo = _bandpass(echo, low, high)
    return echo


def mix_mic_lpb(base):
    rirs = _load_rirs(rir_dir)
    os.makedirs(mix_output_dir, exist_ok=True)
    for _, _, files in os.walk(base):
        for f in files:
            if not f.endswith("-mic.wav"):
                continue
            print(f)
            lpb_f = f.replace("-mic", "-lpb")
            mic, sr1 = sf.read(os.path.join(base, f))
            lpb, sr2 = sf.read(os.path.join(base, lpb_f))
            if mic.ndim > 1:
                mic = mic[:, 0]
            if lpb.ndim > 1:
                lpb = lpb[:, 0]
            if sr1 != fs or sr2 != fs:
                raise ValueError("Expected %d Hz mono audio." % fs)
            # align lengths
            n = min(len(mic), len(lpb))
            mic, lpb = mic[:n].astype("float32"), lpb[:n].astype("float32")
            # synthesize the acoustic echo from the loopback signal
            echo = synth_echo(lpb, rirs)
            # scale the echo to hit a random signal-to-echo ratio, then mix
            ser_db = np.random.uniform(*ser_range_db)
            scale = _rms(mic) / (_rms(echo) + 1e-12) * (10.0 ** (-ser_db / 20.0))
            mixed = mic + scale * echo
            # avoid clipping
            peak = float(np.max(np.abs(mixed)) + 1e-12)
            if peak > 0.99:
                mixed = mixed * (0.99 / peak)
            out_name = f.replace("-mic", "-mix-echo")
            sf.write(os.path.join(mix_output_dir, out_name), mixed.astype("float32"), fs)


mix_mic_lpb("/home/allen/Data/ainur/aec/train_data_aligned/align_wav_lpb_mic/")
