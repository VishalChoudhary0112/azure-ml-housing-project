import joblib
import json
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)

def run(data):
    try:
        input_data = np.array(json.loads(data)["data"])
        result = model.predict(input_data)
        return result.tolist()
    except Exception as e:
        return str(e)
