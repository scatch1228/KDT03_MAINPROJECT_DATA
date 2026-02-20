import time
import json
import sys
from inference import ReservoirInferenceService, PumpOptimizationService
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd

window_size=180
forecast_size=15
total_forecast_size=60
input_dim = 10

configs = {
    4: {"weights": "../model/a_resv_flow_model.pth", 
           "scaler_x": "../model/a_resv_scaler_x.pkl",
           "scaler_y": "../model/a_resv_scaler_y.pkl",
           "config": "../model/a_resv_config.json"
           },
    7: {"weights": "../model/d_resv_flow_model.pth", 
           "scaler_x": "../model/d_resv_scaler_x.pkl",
           "scaler_y": "../model/d_resv_scaler_y.pkl",
           "config": "../model/d_resv_config.json"
           },
    8: {"weights": "../model/e_resv_flow_model.pth", 
            "scaler_x": "../model/e_resv_scaler_x.pkl",
            "scaler_y": "../model/e_resv_scaler_y.pkl",
            "config": "../model/e_resv_config.json"
            },
    10: {"weights": "../model/g_resv_flow_model.pth", 
          "scaler_x": "../model/g_resv_scaler_x.pkl",
          "scaler_y": "../model/g_resv_scaler_y.pkl",
          "config": "../model/g_resv_config.json"
          },
    13: {"weights": "../model/j_resv_flow_model.pth", 
           "scaler_x": "../model/j_resv_scaler_x.pkl",
           "scaler_y": "../model/j_resv_scaler_y.pkl",
           "config": "../model/j_resv_config.json"
           },
    15: {"weights": "../model/l_resv_flow_model.pth", 
           "scaler_x": "../model/l_resv_scaler_x.pkl",
           "scaler_y": "../model/l_resv_scaler_y.pkl",
           "config": "../model/l_resv_config.json"
           }
}
resv_service = ReservoirInferenceService(configs, input_dim=input_dim, window_size=window_size)

def get_mysql_engine():
    try:
        user = 'musthave'
        password = 'tiger'
        host = '10.125.121.184'
        port = 3306
        database = 'pms_db_dev_gs'
        
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        
        engine = create_engine(url)
        return engine
    except Exception as e:
        print(f'MySQL Engine creation error: {e}')
        sys.exit(1)

def get_latest_window(resv: int, start_date):
    engine = get_mysql_engine()
    
    query = text("""
        SELECT 
            r.collected_at, 
            r.flow_out as resv_flow,
            w.temperature,
            w.rainfall as precipitate,
            w.humidity
        FROM reservoir_minutely r
        JOIN weather w ON r.collected_at = w.collected_at
        WHERE r.facility_id = :resv 
        AND r.collected_at >= :start_date
        LIMIT 240
    """)
    
    try: 
        params = {"resv": resv, "start_date": start_date}
        df = pd.read_sql(query, engine, params=params)
        
        if df.empty:
            return None, None, None

        if resv == 4:
            t = df['collected_at']

            minute_of_day = t.dt.hour * 60 + t.dt.minute
            df['time_sin'] = 0.5 * np.sin(2 * np.pi * minute_of_day / 1440) + 0.5
            df['time_cos'] = 0.5 * np.cos(2 * np.pi * minute_of_day / 1440) + 0.5

            dow = t.dt.dayofweek
            df['dow_sin'] = 0.5 * np.sin(2 * np.pi * dow / 7) + 0.5
            df['dow_cos'] = 0.5 * np.cos(2 * np.pi * dow / 7) + 0.5

            doy = t.dt.dayofyear
            df['season_sin'] = 0.5 * np.sin(2 * np.pi * doy / 365.25) + 0.5
            df['season_cos'] = 0.5 * np.cos(2 * np.pi * doy / 365.25) + 0.5

            train_df = df[:-15] # 길이 225
            val_df = df['resv_flow'][-60:].values
            columns = ['resv_flow', 'temperature', 'precipitate','humidity',
                    'time_sin', 'time_cos', 'dow_sin', 'dow_cos', 'season_sin', 'season_cos'
                    ]
        
        else:
            train_df = df[:-15] # 길이 225
            val_df = df['resv_flow'][-60:].values
            columns = ['resv_flow', 'temperature', 'precipitate','humidity',]
        
        return train_df[columns], train_df['collected_at'].values[-1], val_df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None
    finally:
        engine.dispose()


def format_to_json(prediction, last_input_time, val_df):
    last_input_time = pd.to_datetime(last_input_time)
    prediction = prediction.flatten()

    accuracy = np.mean( (1- np.abs(val_df - prediction) / (val_df+0.01)) )

    json_for_redis = json.dumps(prediction.tolist())
    date_for_redis = str(last_input_time)
    accuracy_for_redis = str(accuracy)

    print(json_for_redis)

    return json_for_redis, date_for_redis, accuracy_for_redis

def run_generator(suzy:int, start_date):
    print(f"Monitoring {suzy}-reservoir flow...")
    print(f"Cycle started at {time.ctime()}")
    try:  
        input_window, last_input_time, val_df= get_latest_window(suzy, start_date)
        prediction = resv_service.predict(suzy, input_window[:180])
        for i in range(1,total_forecast_size//forecast_size):
            prediction = np.concatenate(( prediction, resv_service.predict(suzy,input_window[i*forecast_size : window_size + i*forecast_size]) ))

        json_pred, json_date, json_accuracy = format_to_json(prediction, last_input_time, val_df)

        print(f"Prediction from {json_date}")
        print(f"Cycle complete at {time.ctime()}")

        return json_pred, json_date, json_accuracy
        
    except Exception as e:
        print(f"Error: {e}")

#=======================Pump=========================
#=======================Pump=========================
#=======================Pump=========================
pump_service = PumpOptimizationService()

def get_pump_input(start_time):
    engine = get_mysql_engine()
    # --- 메타 정보 ---
    try:
        resv_info = pd.read_sql(
            """
            SELECT i.facility_id,
                i.area AS estimated_area,
                i.min_level AS safety_min,
                i.max_level AS safety_max
            FROM reservoir_info i
            WHERE i.area > 0
            """,
            engine
        )
        if resv_info.empty:
            return None, None

        valid_ids = resv_info['facility_id'].tolist()

        # --- 시계열 데이터 ---
        query_string = f"""
            SELECT r.collected_at AS timestamp,
                r.facility_id,
                r.level,
                r.flow_out,
                t.press_out_1,
                t.press_out_2,
                t.press_out_3,
                t.press_out_4
            FROM reservoir_minutely r
            LEFT JOIN treatment_minutely t
                ON r.collected_at = t.collected_at
                AND t.facility_id = 1
            WHERE r.facility_id IN ({','.join(map(str, valid_ids))})
            AND r.collected_at >= :start_time
            LIMIT 1440
        """
        query = text(query_string)
        params = {"start_time":start_time}
        df_all = pd.read_sql(query, engine, params=params)
        if df_all.empty:
            return None, None

        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])

        # 실제 펌프 가동 대수 (참고용)
        pump_cols = ['press_out_1', 'press_out_2', 'press_out_3', 'press_out_4']
        df_all['actual_pumps'] = (df_all[pump_cols] >= 2.0).sum(axis=1)

        # --- 배수지별 평균 유출량 기반 분배 비율 ---
        avg_outflow = (
            df_all.groupby('facility_id')['flow_out']
            .mean()
            .reset_index()
        )
        avg_outflow['dist_rate'] = avg_outflow['flow_out'] / avg_outflow['flow_out'].sum()

        resv_info = pd.merge(
            resv_info,
            avg_outflow[['facility_id', 'dist_rate']],
            on='facility_id'
        )
        return df_all, resv_info
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None
    finally:
        engine.dispose()
    
def run_optimizer(start_time:str):
    print(f"Optimizing pump config...")
    print(f"Cycle started at {time.ctime()}")
    try:
        df_all, resv_info = get_pump_input(start_time)
        optimization = pump_service.optimize(df_all, resv_info)
        json_result = optimization.to_json(orient='records', date_format='iso')
        return json_result
    
    except Exception as e:
        print(f"Error: {e}")