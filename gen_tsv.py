# Written by ChatGPT 4o
import os
import argparse
import soundfile as sf

def get_audio_length(file_path):
    try:
        with sf.SoundFile(file_path) as f:
            return int(len(f) / f.samplerate * 16000)  # 轉換成 16kHz 長度
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def generate_audio_metadata(root_dir, ext, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{root_dir}\n")  # 第一行寫入 root_dir
        
        for subdir, _, files in os.walk(root_dir):
            for file in sorted(files):
                if file.endswith(ext):
                    file_path = os.path.join(subdir, file)
                    rel_path = os.path.relpath(file_path, root_dir)
                    length = get_audio_length(file_path)
                    if length is not None:
                        f.write(f"{rel_path}\t{length}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio metadata file.")
    parser.add_argument("root_dir", type=str, help="Root directory containing audio files.")
    parser.add_argument("--ext", type=str, default=".flac", help="Audio file extension (default: .flac).")
    parser.add_argument("--output", type=str, default="audio_metadata.txt", help="Output file name.")
    args = parser.parse_args()

    generate_audio_metadata(args.root_dir, args.ext, args.output)
    print(f"Metadata file saved to {args.output}")