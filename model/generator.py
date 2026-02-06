import time
import json
from datetime import timedelta
from inference import ReservoirInferenceService
from redis import Redis
import pymysql
import pandas as pd

# 1. SETUP
# Initialize Inference 
window_size=180
configs = {
    # 4: {"weights": "a_resv/a_resv_flow_model.pth", 
    #       "scaler": "a_resv/a_resv_scaler.pkl",
    #       "config": "a_resv/a_resv_config.json"
    #       },
    # 5: {"weights": "b_resv/b_resv_flow_model.pth", 
    #       "scaler": "b_resv/b_resv_scaler.pkl",
    #       "config": "b_resv/b_resv_config.json"
    #       },
    # 6: {"weights": "c_resv/c_resv_flow_model.pth", 
    #       "scaler": "c_resv/c_resv_scaler.pkl",
    #       "config": "c_resv/c_resv_config.json"
    #       },
    # 7: {"weights": "d_resv/d_resv_flow_model.pth", 
    #       "scaler": "d_resv/d_resv_scaler.pkl",
    #       "config": "d_resv/d_resv_config.json"
    #       },
    # 8: {"weights": "e_resv/e_resv_flow_model.pth", 
    #       "scaler": "e_resv/e_resv_scaler.pkl",
    #       "config": "e_resv/e_resv_config.json"
    #       },
    # 9: {"weights": "f_resv/f_resv_flow_model.pth", 
    #       "scaler": "f_resv/f_resv_scaler.pkl",
    #       "config": "f_resv/f_resv_config.json"
    #       },
    10: {"weights": "g_resv/g_resv_flow_model.pth", 
          "scaler": "g_resv/g_resv_scaler.pkl",
          "config": "g_resv/g_resv_config.json"
          },
    # 11: {"weights": "h_resv/h_resv_flow_model.pth", 
    #       "scaler": "h_resv/h_resv_scaler.pkl",
    #       "config": "h_resv/h_resv_config.json"
    #       },
    # 12: {"weights": "i_resv/i_resv_flow_model.pth", 
    #       "scaler": "i_resv/i_resv_scaler.pkl",
    #       "config": "i_resv/i_resv_config.json"
    #       },
    # 13: {"weights": "j_resv/j_resv_flow_model.pth", 
    #       "scaler": "j_resv/j_resv_scaler.pkl",
    #       "config": "j_resv/j_resv_config.json"
    #       },
    # 14: {"weights": "k_resv/k_resv_flow_model.pth", 
    #       "scaler": "k_resv/k_resv_scaler.pkl",
    #       "config": "k_resv/k_resv_config.json"
    #       },
    # 15: {"weights": "l_resv/l_resv_flow_model.pth", 
    #       "scaler": "l_resv/l_resv_scaler.pkl",
    #       "config": "l_resv/l_resv_config.json"
    #       }
}
service = ReservoirInferenceService(configs,window_size=window_size)
#redis_client = Redis(host="10.125.121.184", port=6379, decode_responses=True)

def connect_MySQL():
    return pymysql.connect(
        host='10.125.121.184',
        port=3306,
        user='musthave',
        password='tiger',
        database='pms_db_dev_gs',
        charset='utf8mb4',
    )

def get_latest_window(resv: int, start_date):

    connection = connect_MySQL()
    try: 
        query = """
            SELECT collected_at, flow_out
            FROM reservoir_minutely 
            WHERE facility_id = %s 
            AND collected_at >= %s
            AND collected_at < DATE_ADD(%s, INTERVAL 3 HOUR)
            ORDER BY collected_at ASC 
        """
        df = pd.read_sql(query, connection, params=(resv, start_date, start_date))
        return df['flow_out'].values, df['collected_at'].values[-1]
    finally:
        connection.close()

def format_to_json(prediction, last_input_time):
    last_input_time = pd.to_datetime(last_input_time)
    start_pred_time = last_input_time + timedelta(minutes=1)
    prediction = prediction.reshape(-1,)

    forecast_time = pd.date_range(
        start=start_pred_time,
        periods=len(prediction),
        freq='1min'
        )
    
    pred_df = pd.DataFrame({
        'time': forecast_time,
        'resv_flow_pred':prediction
    })

    # 1. Convert the DataFrame rows to a list of dictionaries
    json_data = pred_df.to_json(orient='records', date_format='iso')
    # 2. Parse that string back to a Python object and wrap it
    # (This step is necessary if you want the 'predictions' header)
    final_dict = {
        "resv_flow_pred": json.loads(json_data)
    }

    # 3. Convert the final dictionary to a JSON string for Redis
    json_for_redis = json.dumps(final_dict)
    date_for_redis = json.dumps(str(last_input_time))

    print(json_for_redis)

    return json_for_redis, date_for_redis

# 2. THE LOOP
def run_generator(task_id:int, suzy:int):
    print("Generator started. Monitoring reservoir flow...")
    while True:
        try:
            # for res_id in configs.keys():
            #     # Get data
            #     input_window, last_input_time= get_latest_window(res_id, start_date="2024-01-01 00:01")
                
            #     # Predict
            #     prediction = service.predict(res_id, input_window)
                
            #     # Push to Redis
            #     json_pred, json_date = format_to_json(prediction, last_input_time)
            #     redis_client.set(f"{res_id}_prediction", json_pred)
            #     redis_client.set(f"{res_id}_last_updated", json_date)
            
            input_window, last_input_time= get_latest_window(10, start_date="2024-01-01 00:01")
            prediction = service.predict(10, input_window)

            json_pred, json_date = format_to_json(prediction, last_input_time)
            #redis_client.set(f"{configs[suzy]}_prediction", json_pred)
            #redis_client.set(f"{configs[suzy]}_last_updated", json_date)

            print(f"Cycle started at {json_date}")
            print(f"Cycle complete at {time.ctime()}")
            time.sleep(60) # Wait for the next minute

            return json_pred, json_date
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10) # Short wait before retry

#run_generator()