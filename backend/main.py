#!/usr/bin/env python3
import os
import json
import argparse
from dotenv import load_dotenv

from open_smile_processor import OpenSmileProcessor
from feature_summarizer import FeatureSummarizer
from llm_analyzer import LLMAnalyzer


def main(audio_file: str, transcript_file: str):
    # 1) Extract acoustic functionals
    smile = OpenSmileProcessor()
    df = smile.extract_features(audio_file)

    # 2) Summarize features
    summarizer = FeatureSummarizer()
    features = summarizer.summarize(df)
    if not features:
        print("No features extracted—check your audio or config.")
        return

    # 3) Load transcript JSON
    if not os.path.exists(transcript_file):
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
    with open(transcript_file, "r") as jf:
        transcript = json.load(jf)
        if not isinstance(transcript, list):
            raise ValueError("Transcript JSON must be a list of segments")

    # 4) Call LLM for combined analysis
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file.")
    analyzer = LLMAnalyzer(api_key)
    summary = analyzer.analyze(features, transcript)

    # 5) Print result
    print("\n=== Combined Tone & Content Summary ===")
    print(summary)
    return summary  # 6) Return features and summary

def runAnalysis(audio_file: str, transcript_file: str):
    # 1) Extract acoustic functionals
    smile = OpenSmileProcessor()
    df = smile.extract_features(audio_file)

    # 2) Summarize features
    summarizer = FeatureSummarizer()
    features = summarizer.summarize(df)
    if not features:
        print("No features extracted—check your audio or config.")
        return
    
    # 4) Call LLM for combined analysis
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file.")
    analyzer = LLMAnalyzer(api_key)
    summary = analyzer.analyze(features, transcript_file)

    # 5) Print result
    print("\n=== Combined Tone & Content Summary ===")
    print(summary)
    return summary  # 6) Return features and summary


if __name__ == "__main__":
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Sales-call tone + transcript analysis"
    )
    p.add_argument("audio_file",   help="Path to WAV file for acoustic analysis")
    p.add_argument(
        "--transcript",
        required=True,
        help="Path to JSON transcript file (speaker-labeled)"
    )
    args = p.parse_args()
    main(args.audio_file, args.transcript)