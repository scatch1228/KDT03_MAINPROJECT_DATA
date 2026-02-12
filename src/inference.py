import torch
import joblib
import numpy as np
import pandas as pd
import json
from flowpredictor import FlowPredictor
from scipy.signal import savgol_filter
#=========Resv Service========
#=========Resv Service========
#=========Resv Service========
class ReservoirInferenceService:
    def __init__(self, reservoir_configs, input_dim, window_size=180):
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
                input_dim=input_dim,
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
        raw_data.loc[:,'resv_flow'] = savgol_filter(raw_data['resv_flow'], window_length=31, polyorder=1)
        scaled_data = scaler_x.transform(raw_data)
        input_tensor = torch.FloatTensor(scaled_data).view(1, n_min, 4)
        
        # 2. Predict
        with torch.no_grad():
            prediction = model(input_tensor)
            
        # 3. Inverse Scale
        out = scaler_y.inverse_transform(prediction.cpu().numpy())
        return out


#=========Pump Service========
#=========Pump Service========
#=========Pump Service========
class PumpOptimizationService:
    def __init__(self):
        self.PUMP_PERFORMANCE = {1: 516.0, 2: 817.0, 3: 1057.0}
        self.PUMP_POWER_KW = 150
        pass

    def optimize(self, df, info_df):
        results = []
        timestamps = sorted(df['timestamp'].unique())

        # 초기 수위
        current_levels = {
            rid: df[df['facility_id'] == rid]['level'].iloc[0]
            for rid in info_df['facility_id']
        }

        current_pumps = 2
        last_change_time = -60
        MIN_HOLDING_TIME = 60

        for i, ts in enumerate(timestamps):
            load_type, price = self.get_load_type(ts)
            curr_rows = df[df['timestamp'] == ts]

            # --- 위험 판단 ---
            danger_low = False
            all_full = True

            for _, res in info_df.iterrows():
                lvl = current_levels[res['facility_id']]
                if lvl <= res['safety_min'] + 0.2:
                    danger_low = True
                if lvl < res['safety_max'] - 0.3:
                    all_full = False
            
            # --- 펌프 대수 제어 (개선 버전) ---
            if (i - last_change_time >= MIN_HOLDING_TIME) or danger_low:
                # 1단계: 시간대별 기본 권장 대수 설정
                if danger_low:
                    target_pumps = 3
                elif load_type == "LOW":
                    target_pumps = 3 if not all_full else 1
                elif load_type == "HIGH":
                    target_pumps = 1
                else:
                    target_pumps = 2

                # 2단계 [핵심]: 만수위 배수지 상황에 따른 강제 하향 조절 (Spill 방지)
                # 현재 모든 배수지가 받아줄 수 있는 유량의 합(dist_rate 총합) 확인
                total_acceptance_ratio = sum(res['dist_rate'] for _, res in info_df.iterrows() 
                                            if current_levels[res['facility_id']] < res['safety_max'] - 0.05)
                
                # 만약 받아줄 배수지가 거의 없다면(예: 20% 미만), 펌프를 최소화(1대)
                if total_acceptance_ratio < 0.2 and not danger_low:
                    new_pumps = 1
                else:
                    new_pumps = target_pumps

                # 최종 가동 대수 결정
                if new_pumps != current_pumps:
                    current_pumps = new_pumps
                    last_change_time = i

            # =========================================================
            # 펌프 이론 유량
            theoretical_inflow_min = self.PUMP_PERFORMANCE[current_pumps] / 60

            # 만수위가 아닌 배수지 & dist_rate 합
            active_resvs = []
            sum_active_dist_rate = 0.0

            for _, res in info_df.iterrows():
                if current_levels[res['facility_id']] < res['safety_max']:
                    active_resvs.append(res['facility_id'])
                    sum_active_dist_rate += res['dist_rate']

            # 실제 정수장 유출량 (받아줄 수 있는 만큼만)
            actual_inflow_min = theoretical_inflow_min * sum_active_dist_rate
            # =========================================================

            spill = theoretical_inflow_min - actual_inflow_min

            # --- 수위 업데이트 ---
            for _, res in info_df.iterrows():
                f_id = res['facility_id']

                # 배수지 유출량
                row = curr_rows[curr_rows['facility_id'] == f_id]
                q_out = row['flow_out'].values[0] / 60 if not row.empty else 0

                # 유입량 계산
                if f_id in active_resvs and sum_active_dist_rate > 0:
                    adjusted_rate = res['dist_rate'] / sum_active_dist_rate
                    q_in = actual_inflow_min * adjusted_rate
                else:
                    q_in = 0

                new_level = current_levels[f_id] + (q_in - q_out) / res['estimated_area']

                # 수위 상하한 강제
                current_levels[f_id] = max(
                    0.1,
                    min(new_level, res['safety_max'])
                )

            results.append({
                'timestamp': ts,
                'active_pumps': current_pumps,
                'sim_levels': current_levels.copy(),
                'sim_cost': (current_pumps * self.PUMP_POWER_KW / 60) * price,
                'spill_m3_per_min': spill
            })

        return pd.DataFrame(results)

    def get_load_type(self, ts):
        #winder
        hour = ts.hour
        if 23 <= hour or hour < 9:
            return "LOW", 70.0
        elif (10 <= hour < 12) or (17 <= hour < 20) or (22 <= hour < 23):
            return "HIGH", 200.0
        else:
            return "MID", 130.0