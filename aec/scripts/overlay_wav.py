import os
from pydub import AudioSegment

mix_output_dir = "/home/allen/Data/ainur/aec/train_data_aligned/mixed"


def loudnorm(audio_segment: AudioSegment, db: float) -> AudioSegment:
    _change = db - audio_segment.dBFS
    return audio_segment.apply_gain(_change)


def mix_mic_lpb(base):
    for _, _, fs in os.walk(base):
        for f in fs:
            if f.endswith("-mic.wav"):
                print(f)
                lpb_f = f.replace("-mic", "-lpb")
                mic_audio_segment = AudioSegment.from_wav(os.path.join(base, f))
                lpb_audio_segment = AudioSegment.from_wav(os.path.join(base, lpb_f))
                mix_audio = mic_audio_segment.overlay(lpb_audio_segment, position=0)
                output_file_name = f.replace("-mic", "-mix-bgm-originloudness")
                mix_audio.export(
                    os.path.join(mix_output_dir, output_file_name), format="wav"
                )


mix_mic_lpb("/home/allen/Data/ainur/aec/train_data_aligned/align_wav_lpb_mic/")
