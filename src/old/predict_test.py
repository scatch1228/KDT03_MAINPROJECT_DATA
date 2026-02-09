# 가상환경에서 fastapi, redis 받아야 됨.
# pip install redis

import asyncio
import json
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from redis.asyncio import Redis
from g_resv_flow_model import g_resv_flow_pred_json

app = FastAPI()
# decode_responses=True를 설정해야 데이터를 읽을 때 문자열로 바로 취급됩니다.
redis_client = Redis(host="10.125.121.184", port=6379, decode_responses=True)

# 이 부분에 데이터 집어 넣어서 전송
async def learning_process(task_id: str):
    print(f"[{task_id}] 학습 시작...")

    # 1. Redis Hash 구조로 저장 (key: result:{task_id})
    # 배열은 문자열(JSON)로 변환해서 넣어야 합니다.
    await redis_client.hset(f"result:{task_id}", mapping={
        "prediction_data": g_resv_flow_pred_json(),
        "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # 2. 만료 시간 설정 (예: 1시간 후 삭제)
    await redis_client.expire(f"result:{task_id}", 3600)
    print(f"[{task_id}] Hash 데이터 저장 완료!")

@app.post("/predict/{task_id}")
async def start_predict(task_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(learning_process, task_id)
    return {"status" : "started", "task_id" : task_id}

# uvicorn predict_test:app --host 0.0.0.0 --port 8000 --reload