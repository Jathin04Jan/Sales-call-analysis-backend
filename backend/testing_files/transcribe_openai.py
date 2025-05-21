"""
Transcribe an audio file using OpenAI Whisper (via openai-python v1) and save the transcript as JSON.
"""
import os
import argparse
import json
from dotenv import load_dotenv
from openai import OpenAI


def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment (.env)")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using Whisper and save JSON transcript."
    )
    parser.add_argument(
        "audio_file", type=str,
        help="Path to input audio file (WAV, MP3, etc.)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to output JSON file (defaults to <audio_filename>.json)"
    )
    args = parser.parse_args()

    audio_path = args.audio_file
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        base, _ = os.path.splitext(audio_path)
        out_path = base + ".json"

    # Call Whisper transcription endpoint
    with open(audio_path, "rb") as af:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=af
        )

    # Build transcript dict (only available attributes)
    transcript = {
        "text": response.text,
        "model": "whisper-1"
    }

    # Save JSON
    with open(out_path, "w") as jf:
        json.dump(transcript, jf, indent=2)

    print(f"Transcription saved to {out_path}")


if __name__ == "__main__":
    main() 