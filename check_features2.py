import joblib
features = joblib.load('models/original_feature_names.pkl')
for i, f in enumerate(features):
    print(i, f)
