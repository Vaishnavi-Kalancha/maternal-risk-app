
import joblib

features = joblib.load("feature_order.pkl")
print("TetanusDose features found in model:")
print([f for f in features if "TetanusDose" in f])

