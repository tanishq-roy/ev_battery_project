import joblib
import pandas as pd
import argparse
import json

def run_inference(input_data):
    try:
        model = joblib.load('models/rf_pipeline.pkl')
    except FileNotFoundError:
        return {"error": "Model not found. Please run train_sklearn.py."}
        
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = pd.DataFrame(input_data)
        
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference via CLI")
    parser.add_argument('--input', type=str, required=True, help="JSON string of input features")
    args = parser.parse_args()
    
    try:
        data = json.loads(args.input)
        result = run_inference(data)
        print(json.dumps(result, indent=2))
    except json.JSONDecodeError:
        print("Error: Input must be a valid JSON string.")
