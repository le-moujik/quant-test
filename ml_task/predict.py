import joblib
import pandas as pd


OUTPUT_PATH = "ml_task/predictions.csv"


def main():
    try:
        df = pd.read_csv("ml_task/hidden_test.csv")
    except FileNotFoundError as e:
        print(e); exit()
    try:
        pipeline = joblib.load("ml_task/model.joblib")
    except FileNotFoundError as e:
        print(e); exit()
    predictions = pipeline.predict(df[['6', '7']].values)
    # predictions = df['7'] + df['6']**2
    predictions = pd.DataFrame({"pred": predictions})
    predictions.to_csv(OUTPUT_PATH)
    print(f"predictions saved to {OUTPUT_PATH}")
    
    
main()
