import ffmpeg
import json
import os
from tqdm import tqdm
import subprocess
from pathlib import Path

def get_total_frames(video_path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=nb_read_frames',
        '-print_format', 'default=nokey=1:noprint_wrappers=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(result.stdout.strip())

def extract_n_frames(video_path, output_dir, n=10):
    os.makedirs(output_dir, exist_ok=True)
    total_frames = get_total_frames(video_path)
    interval = total_frames // n

    for i in range(n):
        frame_number = i * interval
        output_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        (
            ffmpeg
            .input(video_path)
            .filter('select', f'eq(n,{frame_number})')
            .output(output_path, vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )

if __name__=='__main__':
    idx = 1
    dataset = ['msvd', 'msrvtt'][idx]
    ext = ['avi', 'mp4'][idx]

    annotation_path = f"annotations/{dataset}/test_captions.json"
    annotations = json.load(open(annotation_path, 'r'))

    for video in tqdm(annotations):
        video_path = f'annotations/{dataset}/videos/{video}.{ext}'
        output_dir = f'annotations/{dataset}/frames/{video}'
        extract_n_frames(video_path, output_dir, n=5)
