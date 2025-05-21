from pyannote.audio import Pipeline
import os

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("PYANNOTE_AUTH_TOKEN")
)

diarization = pipeline("test2.wav")
print(diarization)