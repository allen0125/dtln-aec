import os
from pydub import AudioSegment


origin_path = "/Users/allen/Project/YaShi/aec/data/origin_data/wav"
align_output_path = "/Users/allen/Project/YaShi/aec/data/origin_data/align_wav"
align_mic_output_path = "/Users/allen/Project/YaShi/aec/data/origin_data/align_wav/mic"
align_lpb_output_path = "/Users/allen/Project/YaShi/aec/data/origin_data/align_wav/lpb"


def align_mic_lpb(base, len_of_samples):
    for _, _, fs in os.walk(base):
        tail_mic_audio = None
        tail_lpb_audio = None
        for f in fs:
            if f.endswith("-mic.wav"):
                real_name = f.split("-")[0]
                lpb_f = f.replace("-mic", "-lpb")
                mic_audio_segment = AudioSegment.from_wav(os.path.join(base, f))
                lpb_audio_segment = AudioSegment.from_wav(os.path.join(base, lpb_f))
                if tail_mic_audio and tail_lpb_audio:
                    mic_audio_segment = tail_mic_audio + mic_audio_segment
                    lpb_audio_segment = tail_lpb_audio + lpb_audio_segment

                if len(mic_audio_segment) > len(lpb_audio_segment):
                    mic_audio_segment = mic_audio_segment[: len(lpb_audio_segment)]
                elif len(mic_audio_segment) < len(lpb_audio_segment):
                    lpb_audio_segment = lpb_audio_segment[: len(mic_audio_segment)]

                len_tail_frame = len(mic_audio_segment) % len_of_samples
                if len_tail_frame:
                    tail_mic_audio = mic_audio_segment[-len_tail_frame:]
                    tail_lpb_audio = lpb_audio_segment[-len_tail_frame:]
                for i in range(len(mic_audio_segment) // len_of_samples):
                    mic_output = mic_audio_segment[
                        i * len_of_samples : (i + 1) * len_of_samples
                    ]
                    lpb_output = lpb_audio_segment[
                        i * len_of_samples : (i + 1) * len_of_samples
                    ]
                    mic_output.export(
                        os.path.join(
                            align_mic_output_path, real_name + f"-aligned-{i}-mic.wav"
                        ),
                        format="wav",
                    )
                    lpb_output.export(
                        os.path.join(
                            align_lpb_output_path, real_name + f"-aligned-{i}-lpb.wav"
                        ),
                        format="wav",
                    )
                    print(f"{real_name}-aligned-{i}")


align_mic_lpb(origin_path, 30000)
