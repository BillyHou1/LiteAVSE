# Fan
# Generate JSON file lists for LRS2.
# LRS2 has official split files (train.txt, val.txt, test.txt),
# use those instead of splitting yourself.
# Each entry is a .mp4 with both audio and video inside.
# This script also extracts audio from each .mp4 using ffmpeg (once, at preprocessing time)
# and saves it as a .wav alongside the .mp4. The dataloader then reads the .wav directly.
# Output: data/lrs_train.json, data/lrs_valid.json, data/lrs_test.json
# Format: [{"audio": "/abs/path/to/utterance.wav", "video": "/abs/path/to/utterance.mp4"}, ...]

import os
import json
import argparse


def read_split_file(split_txt, lrs2_root):
    """
    Read an official LRS2 split file and resolve to absolute paths.

    Args:
        split_txt: str, path to split file (e.g. train.txt)
        lrs2_root: str, root directory of LRS2
    Returns:
        list of {"video": "/abs/path/to/utterance.mp4"}
    """
    # TODO
    raise NotImplementedError


def save_json(data, path):
    """Save a list to a JSON file."""
    # TODO
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Generate LRS2 JSON lists')
    parser.add_argument('--lrs2_root', required=True, help='root directory of LRS2')
    parser.add_argument('--output_dir', default='data', help='output directory for JSON files')
    args = parser.parse_args()

    # TODO read split files, save JSONs, print counts
    raise NotImplementedError


if __name__ == '__main__':
    main()
