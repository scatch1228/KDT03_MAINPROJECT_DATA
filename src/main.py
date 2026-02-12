# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import asyncio
import socket
import json
import sys
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from generator import run_generator, run_optimizer

app = FastAPI()

RESULT_EXPIRE_SECONDS = 3600

configs={
    4:'a',
    7:'d',
    8:'e',
    10:'g',
    13:'j',
    15:'l',
}

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

#Check Redis Connection
#Check Redis Connection
#Check Redis Connection
def is_redis_available(host: str, port: int, timeout: int = 1) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

#Initialize Redis Client
#Initialize Redis Client
#Initialize Redis Client
redis_client = None
_redis_host = "10.125.121.184"
_redis_port = 6379

if is_redis_available(_redis_host, _redis_port):
    redis_client = Redis(host=_redis_host, port=_redis_port, decode_responses=True)
else:
    print(f"[WARNING] Redis 서버에 연결할 수 없습니다: {_redis_host}:{_redis_port}")
    sys.exit(1)


#Saving Logic
#Saving Logic
#Saving Logic
async def _save_result(
    task_id: str,
    status: str,
    task_type: str | None = None,    
    data: str | None = None,
    start_from: str | None = None,
    accuracy: str | None = None,
    error: str | None = None,
):
    if redis_client is None:
        print(f"[{task_id}] Redis 미연결로 결과 저장 불가 (status={status})")
        return
    mapping = {"status": status}

    if task_type == "resv":
        if data is not None:
            mapping["prediction_data"] = data
        if start_from is not None:
            mapping["predict_from"] = start_from
        if accuracy is not None:
            mapping["accuracy"] = accuracy

    elif task_type == "pump":
        if data is not None:
            mapping["optimization_data"] = data
        if start_from is not None:
            mapping["start_from"] = start_from
            
    if error is not None:
        mapping["error"] = error
    key = f"result:{task_id}"
    await redis_client.hset(key, mapping=mapping)
    await redis_client.expire(key, RESULT_EXPIRE_SECONDS)


#Call Prediction Logic
#Call Prediction Logic
#Call Prediction Logic
async def resv_pred(task_id: str, suzy: int):
    print(f"[{task_id}] 학습 시작... ")
    print(f"[배수지: {suzy}]")

    if redis_client is None:
        print(f"[{task_id}] Redis 미연결로 작업 중단")
        await _save_result(task_id, "error", error="Redis 서버 연결 불가")
        return

    if suzy not in configs.keys():
        print(f"[{task_id}] unknown Suzy: {suzy}")
        await _save_result(task_id, "failed", error=f"unknown Suzy: {suzy}")
        return

    try:
        data, time, accuracy = await asyncio.to_thread(run_generator, suzy)
        await _save_result(
            task_type='resv',
            task_id=task_id, 
            status="completed", 
            data=data, 
            start_from=time, 
            accuracy=accuracy
        )
        print(f"[{task_id}] Hash 데이터 저장 완료!")
    except Exception as e:
        print(f"[{task_id}] 오류: {e}")
        await _save_result(task_id, "error", error=str(e))


#Predict API
#Predict API
#Predict API
@app.get("/predict/{suzy}/{task_id}")
async def start_predict(task_id: str, background_tasks: BackgroundTasks, suzy: int):
    background_tasks.add_task(resv_pred, task_id, suzy)
    return {"status" : "started", "task_id" : task_id, "baeSuzy":suzy}

#Result View API
#Result View API
#Result View API
@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """태스크 결과 조회 (Redis에서 조회)."""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis 서버 연결 불가")
    key = f"result:{task_id}"
    raw = await redis_client.hgetall(key)
    if not raw:
        raise HTTPException(
            status_code=404, detail=f"task_id '{task_id}' 결과 없음 또는 만료됨"
        )
    status = raw.get("status", "unknown")
    out = {"task_id": task_id, "status": status}
    if "prediction_data" in raw:
        out["prediction_data"] = json.loads(raw["prediction_data"])
    if "predict_from" in raw:
        out["predict_from"] = raw["predict_from"]
    if "accuracy" in raw:
        out["accuracy"] = raw["accuracy"]
    if "error" in raw:
        out["error"] = raw["error"]
    return out

#Pump Optimization API
#Pump Optimization API
#Pump Optimization API
@app.get("/optimize/{task_id}")
async def start_optimize(task_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(pump_optimizer, task_id)
    return {"status" : "started", "task_id" : task_id, "type":"pump_optimizer"}

async def pump_optimizer(task_id:str, start_time="2024-01-02 00:01:00"):
    print(f"[{task_id}] 최적화 시작... ")
    if redis_client is None:
        print(f"[{task_id}] Redis 미연결로 작업 중단")
        await _save_result(task_id, "error", error="Redis 서버 연결 불가")
        return

    try:
        data = await asyncio.to_thread(run_optimizer, start_time)
        await _save_result(
            task_type='pump',
            task_id=task_id, 
            status="completed", 
            data=data, start_from=start_time
        )
        print(f"[{task_id}] Hash 데이터 저장 완료!")
    except Exception as e:
        print(f"[{task_id}] 오류: {e}")
        await _save_result(task_id, "error", error=str(e))

#Result View API
#Result View API
#Result View API
@app.get("/getopt/{task_id}")
async def get_result(task_id: str):
    """태스크 결과 조회 (Redis에서 조회)."""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis 서버 연결 불가")
    key = f"result:{task_id}"
    raw = await redis_client.hgetall(key)
    if not raw:
        raise HTTPException(
            status_code=404, detail=f"task_id '{task_id}' 결과 없음 또는 만료됨"
        )
    status = raw.get("status", "unknown")
    out = {"task_id": task_id, "status": status}
    if "optimization_data" in raw:
        out["optimization_data"] = json.loads(raw["optimization_data"])
    if "start_from" in raw:
        out["start_from"] = raw["start_from"]
    if "error" in raw:
        out["error"] = raw["error"]
    return out