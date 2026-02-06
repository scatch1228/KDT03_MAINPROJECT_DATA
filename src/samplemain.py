# uvicorn FastAPI:app --host 0.0.0.0 --port 8000 --reload

import asyncio
import json
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from redis.asyncio import Redis
from g_resv.g_resv_flow_model import g_resv_flow_pred_json
from i_resv.i_resv_flow_model import i_resv_flow_pred_json
from inference import ReservoirInferenceService

app = FastAPI()

# Initialize Inference 
configs = {
    "g": {"weights": "g_resv/g_resv_flow_model.pth", "scaler": "g_resv/g_resv_scaler.pkl"}
}
inference_service = ReservoirInferenceService(configs)

# decode_responses=True를 설정해야 데이터를 읽을 때 문자열로 바로 취급됩니다.
redis_client = Redis(host="10.125.121.184", port=6379, decode_responses=True)

#테스트 데이터 생성 및 전송
async def learning_process(task_id: str, suzy: int):
    print(f"[{task_id}] 학습 시작...")
    print(f"[배수지: {suzy}]")
    
    if suzy==10:
        data = g_resv_flow_pred_json()
    elif suzy==12:
        data = i_resv_flow_pred_json()
    else:
        print(f"unknown Suzy:{suzy}")
        return

    # 1. Redis Hash 구조로 저장 (key: result:{task_id})
    # 배열은 문자열(JSON)로 변환해서 넣어야 합니다.
    await redis_client.hset(f"result:{task_id}", mapping={
        "suzy": suzy,
        "prediction_data": data[0],
        "predict_time": data[1]
    })

    # 2. 만료 시간 설정 (예: 1시간 후 삭제)
    await redis_client.expire(f"result:{task_id}", 3600)
    print(f"[{task_id}] Hash 데이터 저장 완료!")

#Demo API
#Demo API
#Demo API
@app.get("/predict/{suzy}/{task_id}")
async def start_predict(task_id: str, background_tasks: BackgroundTasks, suzy: int):
    background_tasks.add_task(learning_process, task_id, suzy)
    return {"status" : "started", "task_id" : task_id, "baeSuzy":suzy}


@app.get("/predict/{suzy}/{task_id}")
async def start_predict(suzy: str, task_id: str, background_tasks: BackgroundTasks):
    data = get_latest_data(suzy)
    prediction = inference_service.predict(suzy, data)
    # ... push to Redis

