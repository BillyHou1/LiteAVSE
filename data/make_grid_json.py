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


def _speaker_from_relpath(rel, parts):
    """
    Infer speaker_id from relative path. Supports:
    - Layout A: grid_root/<speaker>/audio/*.wav  -> rel e.g. "s1/audio", speaker = s1
    - Layout B: grid_root/audio/<speaker>/*.wav  -> rel e.g. "audio/s1", speaker = s1
    """
    if "audio" in parts:
        i = parts.index("audio")
        return parts[i - 1] if i > 0 else (parts[i + 1] if i + 1 < len(parts) else None)
    if "video" in parts:
        i = parts.index("video")
        return parts[i - 1] if i > 0 else (parts[i + 1] if i + 1 < len(parts) else None)
    # Layout A: first part is speaker (e.g. s1/audio)
    if len(parts) >= 2 and parts[1] in ("audio", "video"):
        return parts[0]
    return None


def collect_pairs(grid_root):
    """
    Walk the GRID corpus directory and pair audio (.wav) with video (.mpg)
    by utterance ID for each speaker.

    Supports two structures:
    - Layout A: grid_root/<speaker_id>/audio/*.wav, grid_root/<speaker_id>/video/*.mpg
    - Layout B: grid_root/audio/<speaker_id>/*.wav, grid_root/video/<speaker_id>/*.mpg
    Pairs by same (speaker_id, basename stem), e.g. s1bbaa2p.wav <-> s1bbaa2p.mpg.

    Args:
        grid_root: str, root directory of the GRID corpus
    Returns:
        dict mapping speaker_id (str) -> list of {"audio": ..., "video": ...}
    """
    grid_root = os.path.abspath(grid_root)
    if not os.path.isdir(grid_root):
        print(f"Error: {grid_root} is not a directory")
        print("Expected: <grid_root>/<s1..s34>/audio/*.wav and .../video/*.mpg, or <grid_root>/audio/<s1..s34>/*.wav and .../video/.../*.mpg.")
        print("Please check the path and try again.")
        return {}

    audio_by_key = {}   # (speaker_id, stem) -> abspath
    video_by_key = {}   # (speaker_id, stem) -> abspath

    for dirpath, _dirnames, filenames in os.walk(grid_root):
        rel = os.path.relpath(dirpath, grid_root)
        parts = rel.split(os.sep)
        speaker_id = _speaker_from_relpath(rel, parts)
        if speaker_id is None or speaker_id in ("audio", "video"):
            continue

        for f in filenames:
            stem, ext = os.path.splitext(f)
            ext_lower = ext.lower()
            full = os.path.abspath(os.path.join(dirpath, f))
            key = (speaker_id, stem)
            if ext_lower == ".wav":
                audio_by_key[key] = full
            elif ext_lower == ".mpg":
                video_by_key[key] = full

    pairs_by_speaker = {}
    for key in set(audio_by_key) & set(video_by_key):
        speaker_id, _ = key
        pair = {"audio": audio_by_key[key], "video": video_by_key[key]}
        pairs_by_speaker.setdefault(speaker_id, []).append(pair)

    for spk in list(pairs_by_speaker):
        pairs_by_speaker[spk].sort(key=lambda x: (x["audio"], x["video"]))
    return pairs_by_speaker


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
    train_list = []
    valid_list = []
    test_list = []
    for spk, pairs in pairs_by_speaker.items():
        if spk in train_spk:
            train_list.extend(pairs)
        elif spk in valid_spk:
            valid_list.extend(pairs)
        elif spk in test_spk:
            test_list.extend(pairs)
    return train_list, valid_list, test_list


def save_json(data, path):
    """Save a list to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Generate GRID corpus JSON lists')
    parser.add_argument(
        '--grid_root',
        default=r'C:\Users\Y9000K\Desktop\GRID',
        help='root directory of GRID corpus (default: local GRID path for debug)',
    )
    parser.add_argument('--output_dir', default='data', help='output directory for JSON files')
    args = parser.parse_args()

    # Default GRID split: s1-s28 train, s29-s31 valid, s32-s34 test
    train_spk = [f"s{i}" for i in range(1, 29)]
    valid_spk = [f"s{i}" for i in range(29, 32)]
    test_spk = [f"s{i}" for i in range(32, 35)]

    pairs_by_speaker = collect_pairs(args.grid_root)
    if not pairs_by_speaker:
        _hint = ""
        if os.path.isdir(os.path.abspath(args.grid_root)):
            audio_dir = os.path.join(args.grid_root, "audio")
            if os.path.isdir(audio_dir):
                try:
                    first = next(os.scandir(audio_dir))
                    if first.name.endswith(".tar") or first.name.endswith(".zip"):
                        _hint = " (ERROR: GRID is not unzipped)"
                except StopIteration:
                    pass
        print("No audio/video pairs found. Expected: <grid_root>/<s1..s34>/audio/*.wav and .../video/*.mpg, or <grid_root>/audio/<s1..s34>/*.wav and .../video/.../*.mpg." + _hint)
        return

    train_list, valid_list, test_list = split_by_speaker(
        pairs_by_speaker, train_spk, valid_spk, test_spk
    )

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "grid_train.json")
    valid_path = os.path.join(output_dir, "grid_valid.json")
    test_path = os.path.join(output_dir, "grid_test.json")

    save_json(train_list, train_path)
    save_json(valid_list, valid_path)
    save_json(test_list, test_path)

    print(f"Speakers: {len(pairs_by_speaker)} (train {train_spk[0]}-{train_spk[-1]}, valid {valid_spk[0]}-{valid_spk[-1]}, test {test_spk[0]}-{test_spk[-1]})")
    print(f"Train: {len(train_list)} pairs -> {train_path}")
    print(f"Valid: {len(valid_list)} pairs -> {valid_path}")
    print(f"Test:  {len(test_list)} pairs -> {test_path}")


if __name__ == '__main__':
    main()
