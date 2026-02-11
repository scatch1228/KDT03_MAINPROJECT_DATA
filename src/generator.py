import time
import json
import sys
from datetime import timedelta
from inference import ReservoirInferenceService
from redis import Redis
import pymysql
import numpy as np
import pandas as pd

# 1. SETUP
# Initialize Inference 
window_size=180
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
    # 13: {"weights": "../model/j_resv_flow_model.pth", 
    #        "scaler_x": "../model/j_resv_scaler_x.pkl",
    #        "scaler_y": "../model/j_resv_scaler_y.pkl",
    #        "config": "../model/j_resv_config.json"
    #        },
    15: {"weights": "../model/l_resv_flow_model.pth", 
           "scaler_x": "../model/l_resv_scaler_x.pkl",
           "scaler_y": "../model/l_resv_scaler_y.pkl",
           "config": "../model/l_resv_config.json"
           }
}
service = ReservoirInferenceService(configs,window_size=window_size)
#redis_client = Redis(host="10.125.121.184", port=6379, decode_responses=True)

def connect_MySQL():
    try: 
        connect = pymysql.connect(
        host='10.125.121.184',
        port=3306,
        user='musthave',
        password='tiger',
        database='pms_db_dev_gs',
        charset='utf8mb4',
        )
        return connect
    except OSError:
        print('MySQL connect error')
        sys.exit(1)


def get_latest_window(resv: int, start_date):

    connection = connect_MySQL()
    try: 
        query = """
            SELECT 
            r.collected_at, 
            r.flow_out as resv_flow,
            w.temperature,
            w.rainfall as precipitate,
            w.humidity
            FROM reservoir_minutely r
            JOIN weather w ON r.collected_at = w.collected_at
            WHERE r.facility_id = %s 
            AND r.collected_at >= %s
            LIMIT 195
        """
            #AND r.collected_at < DATE_ADD(%s, INTERVAL 3 HOUR)
            #ORDER BY r.collected_at ASC 
        raw_df = pd.read_sql(query, connection, params=(resv, start_date))#, start_date))
        train_df = raw_df[:-15]
        val_df = raw_df['resv_flow'][-15:].values
        columns = ['resv_flow', 'temperature', 'precipitate','humidity']
        return train_df[columns], train_df['collected_at'].values[-1], val_df
    finally:    
        connection.close()

def format_to_json(prediction, last_input_time, val_df):
    last_input_time = pd.to_datetime(last_input_time)
    start_pred_time = last_input_time + timedelta(minutes=1)
    prediction = prediction.flatten()

    accuracy = np.mean( (1- np.abs(val_df - prediction) / (val_df+0.01)) )
    # forecast_time = pd.date_range(
    #     start=start_pred_time,
    #     periods=len(prediction),
    #     freq='1min'
    #     )
    
    # pred_df = pd.DataFrame({
    #     'time': forecast_time,
    #     'resv_flow_pred':prediction
    # })

    # 1. Convert the DataFrame rows to a list of dictionaries
    #json_data = pred_df.to_json(orient='records', date_format='iso')
    
    # 2. Parse that string back to a Python object and wrap it
    # (This step is necessary if you want the 'predictions' header)
    # final_dict = {
    #     "resv_flow_pred": json.loads(prediction)
    # }

    # 3. Convert the final dictionary to a JSON string for Redis
    json_for_redis = json.dumps(prediction.tolist())
    date_for_redis = str(last_input_time)
    accuracy_for_redis = str(accuracy)

    print(json_for_redis)

    return json_for_redis, date_for_redis, accuracy_for_redis

# 2. THE LOOP
def run_generator(task_id:int, suzy:int):
    print(f"[task_id : {task_id}] started. Monitoring {suzy}-reservoir flow...")
    print(f"Cycle started at {time.ctime()}")
    try:  
        input_window, last_input_time, val_df= get_latest_window(suzy, start_date="2024-01-01 00:01")
        prediction = service.predict(suzy, input_window)

        json_pred, json_date, json_accuracy = format_to_json(prediction, last_input_time, val_df)
        #redis_client.set(f"{configs[suzy]}_prediction", json_pred)
        #redis_client.set(f"{configs[suzy]}_last_updated", json_date)

        print(f"Prediction from {json_date}")
        print(f"Cycle complete at {time.ctime()}")

        return json_pred, json_date, json_accuracy
        
    except Exception as e:
        print(f"Error: {e}")

#run_generator()