# Fan
# Generate JSON file lists for the GRID corpus.
# GRID has 34 speakers (s1-s34) with audio/ and video/ subdirs.
# Pair .wav and .mpg by utterance ID.
# Split: s1-s28 train, s29-s31 valid, s32-s34 test
# Output: data/grid_train.json, data/grid_valid.json, data/grid_test.json
# Format: [{"audio": "/abs/path/...", "video": "/abs/path/..."}, ...]

import os
import json
import argparse


def collect_pairs(grid_root):

    
    """
    Walk the GRID corpus directory and pair audio (.wav) with video (.mpg)
    by utterance ID for each speaker.

    Args:
        grid_root: str, root directory of the GRID corpus
    Returns:
        dict mapping speaker_id (str) -> list of {"audio": ..., "video": ...}
    """
    # TODO
    raise NotImplementedError


def split_by_speaker(pairs_by_speaker, train_spk, valid_spk, test_spk):
    """
    Split the paired data by speaker ID.

    Args:
        pairs_by_speaker: dict from collect_pairs()
        train_spk: list of speaker IDs for training
        valid_spk: list of speaker IDs for validation
        test_spk:  list of speaker IDs for testing
    Returns:
        (train_list, valid_list, test_list) — each a list of dicts
    """
    # TODO
    raise NotImplementedError


def save_json(data, path):
    """Save a list to a JSON file."""
    # TODO
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Generate GRID corpus JSON lists')
    parser.add_argument('--grid_root', required=True, help='root directory of GRID corpus')
    parser.add_argument('--output_dir', default='data', help='output directory for JSON files')
    args = parser.parse_args()

    # TODO collect pairs, split, save, print counts
    raise NotImplementedError


if __name__ == '__main__':
    main()
