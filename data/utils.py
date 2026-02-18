import os
import json

def save_json(data, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def extract_audio_from_video(video_path, sr=16000):
    """
    extract_audio_from_video(video_path), extract audio from video using ffmpeg
    """
    #使用ffmpeg提取音频
    audio_path = video_path.replace('.mp4', '.wav')
    os.system(f'ffmpeg -i {video_path} -vn -ac 1 -ar {sr} {audio_path}')
    return audio_path