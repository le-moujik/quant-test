import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib


MODEL_PATH = "ml_task/model.joblib"


def main():
    try:
        df = pd.read_csv("ml_task/train.csv")
    except FileNotFoundError as e:
        print(e); exit()
    pipeline = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    pipeline.fit(df[['6', '7']].values, df['target'].values)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"model saved to {MODEL_PATH}")
    
    
main()
