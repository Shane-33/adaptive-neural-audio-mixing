# convert_audio.py
import imageio_ffmpeg as ffmpeg
import os

def convert_audio(input_path, output_path):
    """
    Converts all audio files in the input_path to WAV format
    and maintains the directory structure in output_path.
    """
    for root, _, files in os.walk(input_path):
        for file in files:
            # Process only audio files
            if not file.endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.stem.mp4')):
                continue
            
            # Build relative path
            rel_path = os.path.relpath(root, input_path)
            output_dir = os.path.join(output_path, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Construct input and output file paths
            input_file = os.path.join(root, file)
            # Escape special characters
            escaped_file = file.replace(" ", "_").replace("(", "").replace(")", "")
            output_file = os.path.join(output_dir, os.path.splitext(escaped_file)[0] + '.wav')
            
            # Run FFmpeg conversion
            command = f'ffmpeg -i "{input_file}" "{output_file}" -loglevel error'
            result = os.system(command)
            
            if result == 0:
                print(f"Successfully converted: {input_file} -> {output_file}")
            else:
                print(f"Failed to convert: {input_file}")

if __name__ == "__main__":
    INPUT_DIR = "MUSDB18"  # Path to the MUSDB18 dataset with train/test folders
    OUTPUT_DIR = "MUSDB18_wav"  # Path to save the converted dataset
    convert_audio(INPUT_DIR, OUTPUT_DIR)



    