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
        self.scalers_x = {}
        self.scalers_y = {}

        for name, paths in reservoir_configs.items():
            # Load Config (JSON)
            with open(paths['config'], 'r') as f:
                config = json.load(f)

            # Load Scaler
            self.scalers_x[name] = joblib.load(paths['scaler_x'])
            self.scalers_y[name] = joblib.load(paths['scaler_y'])
            
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
        scaler_x = self.scalers_x[reservoir_name]
        scaler_y = self.scalers_y[reservoir_name]
        
        # 1. Scale
        n_min = self.window_size #input window size
        raw_data['resv_flow'] = savgol_filter(raw_data['resv_flow'], window_length=31, polyorder=1)
        scaled_data = scaler_x.transform(raw_data)
        input_tensor = torch.FloatTensor(scaled_data).view(1, n_min, 4)
        
        # 2. Predict
        with torch.no_grad():
            prediction = model(input_tensor)
            
        # 3. Inverse Scale
        out = scaler_y.inverse_transform(prediction.cpu().numpy())
        return out
