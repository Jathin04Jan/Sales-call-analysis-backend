from dotenv import load_dotenv
import os
load_dotenv()
print("HF token:", os.getenv("PYANNOTE_AUTH_TOKEN"))