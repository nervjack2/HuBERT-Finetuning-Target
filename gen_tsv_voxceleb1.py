# Written by ChatGPT 4o
import os
import argparse
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

def get_audio_length(file_path):
    try:
        with sf.SoundFile(file_path) as f:
            return int(len(f) / f.samplerate * 16000)  # 轉換成 16kHz 長度
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def generate_audio_metadata(root_dir, use_list, ext, output_file, index_split):
    usage_list = open(use_list, "r").readlines()
    dataset = []
    for string in tqdm(usage_list):
        pair = string.split()
        index = pair[0]
        if int(index) == index_split:
            x = list(Path(root_dir).glob("*/wav/" + pair[1]))
            dataset.append(str(x[0]))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{root_dir}\n")  # 第一行寫入 root_dir
        for file_path in dataset:
            rel_path = os.path.relpath(file_path, root_dir)
            length = get_audio_length(file_path)
            if length is not None:
                f.write(f"{rel_path}\t{length}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio metadata file.")
    parser.add_argument("root_dir", type=str, help="Root directory containing audio files.")
    parser.add_argument("use_list", type=str, help="Use list of s3prl SID voxceleb1")
    parser.add_argument("--ext", type=str, default=".flac", help="Audio file extension (default: .flac).")
    parser.add_argument("--output", type=str, default="audio_metadata.txt", help="Output file name.")
    parser.add_argument("--index_split", type=int, default=1, help="Dataset split index.")
    args = parser.parse_args()

    generate_audio_metadata(args.root_dir, args.use_list, args.ext, args.output, args.index_split)
    print(f"Metadata file saved to {args.output}")