"""
diarized_transcribe.py

1. Diarize audio with pyannote
2. Transcribe each segment with Whisper
3. Output JSON with speaker labels and text
"""
import os
import argparse
import json
import torch
from dotenv import load_dotenv
from openai import OpenAI
from pyannote.audio import Pipeline
from pydub import AudioSegment

    
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="path to WAV/MP3")
    parser.add_argument("--output", help="path to JSON output")
    args = parser.parse_args()

    audio_file = args.audio_file
    base, _ = os.path.splitext(audio_file)
    out_path = args.output or f"{base}_diarized.json"

    # 1) Speaker diarization
    # You’ll need a Hugging Face token in PYANNOTE_AUTH_TOKEN env var
    #diarization = Pipeline.from_pretrained("pyannote/speaker-diarization",
    #                                       use_auth_token=os.getenv("PYANNOTE_AUTH_TOKEN"))
    
    # pick MPS if available, otherwise CPU
    # choose device
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 1) Speaker diarization
    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=os.getenv("PYANNOTE_AUTH_TOKEN"),
    )
    diarization.to(device)

    # now run inference (this will use MPS if available)
    diar = diarization(audio_file)

    # Load audio with pydub for slicing
    audio = AudioSegment.from_file(audio_file)

    transcript = []
    # 2) For each speech turn, call Whisper
    for turn, _, speaker in diar.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms   = int(turn.end   * 1000)
        segment = audio[start_ms:end_ms]
        # save temp slice
        tmp_path = "_tmp.wav"
        segment.export(tmp_path, format="wav")

        # call Whisper
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        transcript.append({
            "speaker": speaker,
            "start": turn.start,
            "end":   turn.end,
            "text":  resp.text.strip()
        })
        os.remove(tmp_path)

    # 3) Save diarized transcript
    with open(out_path, "w") as jf:
        json.dump(transcript, jf, indent=2)

    print(f"Diarized transcript saved to {out_path}")


def runTranscribe(audio_file: str) -> dict:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    """
    Diarize and transcribe an audio file, returning a JSON object.
    """
    base, _ = os.path.splitext(audio_file)
    out_path = f"{base}_diarized.json"

    # choose device
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 1) Speaker diarization
    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=os.getenv("PYANNOTE_AUTH_TOKEN"),
    )
    diarization.to(device)

    # now run inference (this will use MPS if available)
    diar = diarization(audio_file)


    # Load audio with pydub for slicing
    audio = AudioSegment.from_file(audio_file)

    transcript = []
    # 2) For each speech turn, call Whisper
    for turn, _, speaker in diar.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms   = int(turn.end   * 1000)
        segment = audio[start_ms:end_ms]
        # save temp slice
        tmp_path = "_tmp.wav"
        segment.export(tmp_path, format="wav")

        # call Whisper
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        transcript.append({
            "speaker": speaker,
            "start": turn.start,
            "end":   turn.end,
            "text":  resp.text.strip()
        })
        os.remove(tmp_path)

    # 3) Save diarized transcript
    with open(out_path, "w") as jf:
        json.dump(transcript, jf, indent=2)

    return json.dumps(transcript)  # Return the transcript JSON object
    print(f"Diarized transcript saved to {out_path}")

if __name__ == "__main__":
    main()