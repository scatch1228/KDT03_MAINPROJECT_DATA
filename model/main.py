# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import asyncio
import json
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from generator import run_generator

app = FastAPI()
origins=[
    "http://10.125.121.178:3000",
    "http://10.125.121.184:8080"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = Redis(host="10.125.121.184", port=6379, decode_responses=True)

configs={
    4:'a',
    5:'b',
    6:'c',
    7:'d',
    8:'e',
    9:'f',
    10:'g',
    11:'h',
    12:'j',
    13:'k',
    14:'l',
}

#배수지 예측
async def resv_pred(task_id: str, suzy: int):
    print(f"[{task_id}] 학습 시작...")
    print(f"[배수지: {suzy}]")

    data = run_generator(task_id, suzy)

    # 1. Redis Hash 구조로 저장 (key: result:{task_id})
    # 배열은 문자열(JSON)로 변환해서 넣어야 합니다.
    await redis_client.hset(f"result:{task_id}", mapping={
        "prediction_data": data[0],
        "predict_time": data[1]
    })

    # 2. 만료 시간 설정 (예: 1시간 후 삭제)
    await redis_client.expire(f"result:{task_id}", 600)
    print(f"[{task_id}] Hash 데이터 저장 완료!")

@app.get("/predict/{suzy}/{task_id}")
async def start_predict(task_id: str, background_tasks: BackgroundTasks, suzy: int):
    background_tasks.add_task(resv_pred, task_id, suzy)
    return {"status" : "started", "task_id" : task_id, "baeSuzy":suzy}



