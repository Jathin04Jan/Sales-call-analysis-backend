import opensmile
import pandas as pd

class OpenSmileProcessor:
    """
    Wraps the OpenSMILE Python API to extract functionals from an audio file.
    """
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract_features(self, audio_path: str) -> pd.DataFrame:
        df = self.smile.process_file(audio_path)
        if df.empty:
            print("No data processed from the audio file.")
        return df