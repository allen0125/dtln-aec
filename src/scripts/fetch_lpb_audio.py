import os
import csv
from urllib.parse import urlparse

import requests

audio_track_data = {}
audio_track_data_file_dir = '/Users/allen/Project/YaShi/aec/data/audio_track_info.csv'

lpb_output_dir = '/Users/allen/Project/YaShi/aec/data/origin_data/clear/lpb/'


with open(audio_track_data_file_dir, 'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        audio_track_data[os.path.basename(row[4]).split('.')[0]] = row[2]


def download_mp3(mp3_url, output):
    if os.path.exists(output):
        return True

    r = requests.get(mp3_url, timeout=30)
    with open(output, "wb") as mp3_file:
        mp3_file.write(r.content)
    return True


def fetch_lpbs(base):
    for _, _, fs in os.walk(base):
        for f in fs:
            if f.endswith('.mp3') or f.endswith('.mp4') or f.endswith('.m4a'):
                name = f.split('.')[0]
                lpb_url = ''
                if name in audio_track_data.keys():
                    lpb_url = audio_track_data[name]
                if lpb_url:
                    _output_name = name + '.' + lpb_url.split('.')[-1]
                    _output_dir = os.path.join(lpb_output_dir, _output_name)
                    download_mp3(lpb_url, _output_dir)   
                    print("Success get {}'s lpb file".format(f))


fetch_lpbs('/Users/allen/Project/YaShi/aec/data/origin_data/clear')