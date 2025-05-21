import pandas as pd

class FeatureSummarizer:
    """
    Prepares an LLM-friendly summary dict from OpenSMILE DataFrame.
    """
    DEFAULT_FEATURE_MAP = {
    # Pitch
    'F0semitoneFrom27.5Hz_sma3nz_amean': 'mean_pitch_semitone',
    'F0semitoneFrom27.5Hz_sma3nz_stddev': 'pitch_stddev',
    # Speaking rate proxy
    'voicedSegmentsPerSec_sma3nz_amean': 'voiced_segments_per_sec',
    # Loudness
    'loudness_sma3nz_amean': 'mean_loudness',
    'loudness_sma3nz_stddev': 'loudness_stddev',
    'loudness_sma3nz_min': 'loudness_min',
    'loudness_sma3nz_max': 'loudness_max',
    # Voice quality / noise
    'HNRdBACF_sma3nz_amean': 'mean_hnr',
    # Silence / pauses
    'silenceRate_sma3nz_amean': 'silence_rate',
    # Jitter and shimmer (roughness)
    'jitterLocal_sma3nz_amean': 'jitter_local',
    'shimmerLocal_sma3nz_amean': 'shimmer_local',
    # Spectral/timbre features
    'spectralFlux_sma3nz_amean': 'spectral_flux',
    'spectralCentroid_sma3nz_amean': 'spectral_centroid',
    # Example MFCC; add mfcc2…mfcc13 similarly if desired
    'mfcc1_sma3nz_amean': 'mfcc1_mean',
    }

    def __init__(self, feature_map: dict = None):
        self.feature_map = feature_map or self.DEFAULT_FEATURE_MAP

    def summarize(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            key: float(row[col])
            for col, key in self.feature_map.items()
            if col in row
        }