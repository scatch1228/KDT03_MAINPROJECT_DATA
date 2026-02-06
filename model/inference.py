import torch
import joblib
import numpy as np
import json
from flowpredictor import FlowPredictor
from scipy.signal import savgol_filter
#=========Real Service========
#=========Real Service========
#=========Real Service========
class ReservoirInferenceService:
    def __init__(self, reservoir_configs, window_size=180):
        self.window_size = window_size
        
        # reservoir_configs = {'g': {'weights': 'path', 'scaler': 'path'}, 'i': {...}}
        self.models = {}
        self.scalers = {}

        for name, paths in reservoir_configs.items():
            # Load Config (JSON)
            with open(paths['config'], 'r') as f:
                config = json.load(f)

            # Load Scaler
            self.scalers[name] = joblib.load(paths['scaler'])
            
            # Load Model
            model = FlowPredictor(
                hidden_dim=config['units'],
                output_dim=config['forecast_size'],
                dropout=config['dropout']
            )
            model.load_state_dict(torch.load(paths['weights'], map_location=torch.device('cpu')))
            model.eval()
            self.models[name] = model

    def predict(self, reservoir_name, raw_data):
        # 0. Select the correct tools
        model = self.models[reservoir_name]
        scaler = self.scalers[reservoir_name]
        
        # 1. Scale
        n_min = self.window_size #input window size
        filtered_data = savgol_filter(raw_data, window_length=31, polyorder=1)
        scaled_data = scaler.transform(filtered_data.reshape(-1, 1))
        input_tensor = torch.FloatTensor(scaled_data).view(1, n_min, 1)
        
        # 2. Predict
        with torch.no_grad():
            prediction = model(input_tensor)
            
        # 3. Inverse Scale
        return scaler.inverse_transform(prediction.numpy())
