import joblib

preprocessor = joblib.load('models/ids_preprocessor.pkl')
print(preprocessor.feature_names_in_)
