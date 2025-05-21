import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
df = smile.process_file('test2.wav')
if not df.empty:
    print(df.head())
else:
    print("No data processed from the audio file.")