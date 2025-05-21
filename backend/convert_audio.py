from pydub import AudioSegment
import os

def convert_to_model_format(input_path, output_path):
    """
    Converts input audio to mono-channel, 44100Hz WAV format.
    
    Parameters:
        input_path (str): Path to the input audio file.
        output_path (str): Path where the converted audio will be saved.
    """
    try:
        # Load audio using pydub
        audio = AudioSegment.from_file(input_path)

        # Convert to mono channel and 44100 Hz
        audio = audio.set_channels(1).set_frame_rate(44100)

        # Export as WAV
        audio.export(output_path, format="wav")
        print(f"Converted and saved: {output_path}")

    except Exception as e:
        print(f"Error during conversion: {e}")

# Example usage
if __name__ == "__main__":
    input_file = "test2.mp3"         # Replace with your actual recording
    output_file = "test2.wav"              # This will be the input to the model
    convert_to_model_format(input_file, output_file)