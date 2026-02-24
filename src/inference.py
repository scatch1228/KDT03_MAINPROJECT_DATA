import torch
import joblib
import numpy as np
import pandas as pd
import json
from flowpredictor import FlowPredictor
from flowtransformer import FlowTransformer
from scipy.signal import savgol_filter
#=========Resv Service========
#=========Resv Service========
#=========Resv Service========
class ReservoirInferenceService:
    def __init__(self, reservoir_configs, input_dim, window_size=180):
        self.window_size = window_size
        self.input_dim = input_dim

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
            
            #a,d 배수지만 input_dim = 9 
            if name == 4 or name == 7:
                model = FlowTransformer(
                    input_dim=self.input_dim,
                    d_model=config['d_model'],
                    n_head=config['n_head'],
                    num_layers=config['num_layers'],
                    output_dim=config['forecast_size'],
                    dropout=config['dropout']
                )
            else:
                # Load Model
                model = FlowPredictor(
                    input_dim=4,
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
        input_tensor = torch.FloatTensor(scaled_data).view(1, n_min, model.input_dim)
        
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
        self.PUMP_PERFORMANCE = {1: 425.8, 2: 715.3, 3: 902.5}
        self.PUMP_POWER_KW = 150

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
            ts_pd = pd.Timestamp(ts)
            load_type, price = self.get_load_type(ts)
            curr_rows = df[df['timestamp'] == ts]

            # --- 위험 판단 ---
            danger_low = any(current_levels[rid] <= info_df.loc[info_df['facility_id']==rid, 'safety_min'].values[0] + 0.2 for rid in current_levels)
            
            # --- 펌프 대수 제어 ---
            if (i - last_change_time >= MIN_HOLDING_TIME) or danger_low:
                # 1단계: 시간대별 기본 권장 대수 설정
                if danger_low:
                    new_pumps = 3
                elif load_type == "LOW":
                    target_hour = 9
                    target_end = ts_pd.replace(hour=target_hour, minute=0, second=0)
                    if ts_pd.hour >= target_hour:
                        target_end += pd.Timedelta(days=1)
                    
                    remaining_mins = (target_end - ts_pd).total_seconds() / 60
                    
                    if remaining_mins > 0:
                        total_required_vol = 0
                        # 목표 수위를 max의 95% 정도로 약간 낮춤 (안정적 가동 유도)
                        TARGET_RATIO = 0.85 
                        
                        for _, res in info_df.iterrows():
                            f_id = res['facility_id']
                            # 여유량 계산 시 목표 수위 적용
                            fill_vol = (res['safety_max'] * TARGET_RATIO - current_levels[f_id]) * res['estimated_area']
                            future_demand = df[(df['facility_id'] == f_id) & (df['timestamp'] > ts_pd) & (df['timestamp'] <= target_end)]['flow_out'].sum() / 60
                            total_required_vol += (max(0, fill_vol) + future_demand)
                        
                        required_flow_hr = (total_required_vol / remaining_mins) * 60
                        total_sim_dist_rate = info_df['dist_rate'].sum()

                        # 대수 결정 로직 강화: 2대로도 85% 이상 채울 수 있다면 굳이 3대를 틀지 않음
                        p2_capacity = self.PUMP_PERFORMANCE[2] * total_sim_dist_rate
                        p3_capacity = self.PUMP_PERFORMANCE[3] * total_sim_dist_rate
                        
                        if required_flow_hr > p3_capacity * 0.95: # 정말 모자랄 때만 3대
                            new_pumps = 3
                        elif required_flow_hr > p2_capacity * 0.9: # 2대 성능의 90% 이상 필요할 때 2대
                            new_pumps = 2
                        else:
                            new_pumps = 1
                elif load_type == "HIGH":
                    new_pumps = 1
                else:
                    new_pumps = 2
                if new_pumps != current_pumps:
                    current_pumps = new_pumps
                    last_change_time = i

            # 현재 수위를 고려한 "가변 유입 분배 비율" 계산
            # 수위 여유분(m)이 적은 곳(낮은 곳)에 가중치를 더 부여
            fill_priority = {}
            total_priority = 0
            for _, res in info_df.iterrows():
                f_id = res['facility_id']
                # 만수위까지 남은 높이 (여유량)
                gap = max(0.01, res['safety_max'] - current_levels[f_id])
                # 기본 분배율(dist_rate)에 현재 여유도를 곱해 가중치 생성
                priority = res['dist_rate'] * (gap ** 2) # 제곱을 사용하여 수위가 낮은 곳에 더 강력하게 배분
                fill_priority[f_id] = priority
                total_priority += priority

            # 수위 업데이트 및 유량 물리 계산
            theoretical_inflow_min = self.PUMP_PERFORMANCE[current_pumps] / 60
            active_resvs = [rid for rid, lvl in current_levels.items() if lvl < info_df.loc[info_df['facility_id']==rid, 'safety_max'].values[0]]
            sum_active_priority = sum(fill_priority[rid] for rid in active_resvs)
            
            # 실제 받아주는 유량 (spill 계산용)
            sum_active_dist_rate = info_df[info_df['facility_id'].isin(active_resvs)]['dist_rate'].sum()
            actual_inflow_min = theoretical_inflow_min * sum_active_dist_rate
            spill = theoretical_inflow_min - actual_inflow_min

            for _, res in info_df.iterrows():
                f_id = res['facility_id']
                # 배수지 유출량
                q_out = curr_rows[curr_rows['facility_id'] == f_id]['flow_out'].values[0] / 60 if not curr_rows[curr_rows['facility_id'] == f_id].empty else 0
                
                # 유입량 계산
                if f_id in active_resvs and sum_active_priority > 0:
                    # 수위가 반영된 dynamic_rate 사용
                    dynamic_rate = fill_priority[f_id] / sum_active_priority
                    q_in = actual_inflow_min * dynamic_rate
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