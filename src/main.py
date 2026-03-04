# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import asyncio
import socket
import json
import sys
from time import ctime as ctime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from generator import run_generator, run_optimizer, run_simulator

# --- Pydantic Models for Documentation ---
class PredictResponse(BaseModel):
    status: str = Field(..., description="작업 상태 (started)", example="started")
    task_id: str = Field(..., description="고유 작업 ID", example="8")
    baeSuzy: int = Field(..., description="배수지 번호", example="8")

class PredictionResult(BaseModel):
    task_id: str
    status: str = Field(..., description="작업 상태 (completed/failed/error)", example="completed")
    prediction_data: Optional[List[float]] = Field(None, description="예측된 수요 데이터 리스트",
                                                            example="[23.1, 22.1, 20.1, ...]")
    predict_from: Optional[str] = Field(None, description="예측 시작 시각", 
                                        example="2024-01-01 00:15")
    accuracy: Optional[float] = Field(None, description="모델 예측 정확도 (0~1)",
                                      example="0.82")
    error: Optional[str] = Field(None, description="에러 발생 시 메시지", example="")

class OptimizeResponse(BaseModel):
    status: str = Field(..., example="started")
    task_id: str
    type: str = "pump_optimizer"

class OptimizationResult(BaseModel):
    task_id: str
    status: str = Field(..., description="작업 상태 (completed/error)", example="completed")
    optimization_data: Optional[List[Dict[str, Any]]] = Field(None, description="최적화된 펌프가동 데이터 리스트",
                                                              #example=""
                                                              )
    start_from: Optional[str] = Field(None, description="최적화 시작 시각",
                                      example="2024-01-01 00:00")
    error: Optional[str] = Field(None, description="에러 발생 시 메시지", example="")

# --- App Initialization ---
app = FastAPI(
    title="KDT2-3 메인 프로젝트 데이터 API",
    description = """
    ## 배수지 수요예측 및 펌프 최적화 시스템
    이 API는 두 가지 주요 기능을 제공합니다:
    1. **배수지 수요예측**: 특정 배수지의 향후 1시간 수요를 15분 단위로 예측합니다.
    2. **펌프 최적화**: 예측된 수요를 바탕으로 24시간 펌프 운영 스케줄을 최적화합니다.
    """,
    version = "0.1"
)

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
async def resv_pred(task_id: str, suzy: int, start_date: str = "2024-01-01 00:01"):
    print(f"[{task_id}] 학습 시작... ")
    print(f"[배수지: {suzy}]")
    # 날짜가 인자로 잘 받아지는지 확인
    print(f"start_date: {start_date}")

    if redis_client is None:
        print(f"[{task_id}] Redis 미연결로 작업 중단")
        await _save_result(task_id, "error", error="Redis 서버 연결 불가")
        return

    if suzy not in configs.keys():
        print(f"[{task_id}] unknown Suzy: {suzy}")
        await _save_result(task_id, "failed", error=f"unknown Suzy: {suzy}")
        return

    try:
        data, time, accuracy = await asyncio.to_thread(run_generator, suzy, start_date)
        await _save_result(
            task_type='resv',
            task_id=task_id, 
            status="completed", 
            data=data, 
            start_from=time, 
            accuracy=accuracy
        )
        print(f"[{task_id}] Hash 데이터 저장 완료! at {ctime()}")
    except Exception as e:
        print(f"[{task_id}] 오류: {e}")
        await _save_result(task_id, "error", error=str(e))

#Call Optimization Logic
#Call Optimization Logic
#Call Optimization Logic
async def pump_optimizer(task_id:str, start_time="2024-01-01 00:01"):
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
            data=data, 
            start_from=start_time
        )
        print(f"[{task_id}] Hash 데이터 저장 완료! at {ctime()}")
    except Exception as e:
        print(f"[{task_id}] 오류: {e}")
        await _save_result(task_id, "error", error=str(e))

#Call Simulation Logic
#Call Simulation Logic
#Call Simulation Logic
async def pump_simulator(task_id:str, pump:str, start_time="2024-01-01 00:01"):
    print(f"[{task_id}] 시뮬레이션 시작... ")
    if redis_client is None:
        print(f"[{task_id}] Redis 미연결로 작업 중단")
        await _save_result(task_id, "error", error="Redis 서버 연결 불가")
        return

    try:
        data = await asyncio.to_thread(run_simulator, start_time, int(pump))
        await _save_result(
            task_type='pump',
            task_id=task_id, 
            status="completed", 
            data=data, 
            start_from=start_time
        )
        print(f"[{task_id}] Hash 데이터 저장 완료! at {ctime()}")
    except Exception as e:
        print(f"[{task_id}] 오류: {e}")
        await _save_result(task_id, "error", error=str(e))


#================END POINTS===============
#================END POINTS===============
#================END POINTS===============

#Predict API
#Predict API
#Predict API
@app.get("/predict/{suzy}/{task_id}/{start_date}", 
         tags=["배수지"],
         summary="배수지 수요예측 시작",
         response_model=PredictResponse
         )
async def start_predict(task_id: str, background_tasks: BackgroundTasks, suzy: int, 
                        start_date
                        ):
    """
    요청된 시각의 **15분 시간대 기준**으로 배수지 수요예측을 시작합니다. 
    1시간 예측값이 Redis DB에 전달됩니다. 
    예측값은 Redis에 10분 동안 저장됩니다. 
    """
    background_tasks.add_task(resv_pred, task_id, suzy, start_date)
    return {"status" : "started", "task_id" : task_id, "baeSuzy":suzy}

#Result View API
#Result View API
#Result View API
@app.get("/result/predict/{task_id}", 
         tags=["배수지"],
         summary="수요예측 결과 조회",
        response_model=PredictionResult
         )
async def get_result(task_id: str):
    """
    Redis에 저장된 배수지 수요예측 결과 조회.
    """
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
@app.get("/optimize/{task_id}/{start_time}", 
        tags=["정수장"],
        summary="펌프 운영 최적화 시작",
        response_model=OptimizeResponse 
        )
async def start_optimize(task_id: str, start_time: str, background_tasks: BackgroundTasks):
    """
    24시간 펌프 스케줄링 최적화 알고리즘을 실행합니다.
    배수지별 예상 수위 변화량을 함께 계산합니다. 
    """

    background_tasks.add_task(pump_optimizer, task_id, start_time)
    return {"status" : "started", "task_id" : task_id, "type":"pump_optimizer"}

#Result View API
#Result View API
#Result View API
@app.get("/result/optimize/{task_id}", 
        tags=["정수장"],
        summary="펌프 운영 최적화 결과 조회",
        response_model=OptimizationResult
        )
async def get_result(task_id: str):
    """
    Redis에 저장된 펌프가동 최적화 결과 조회.
    """
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

#Pump Simulation API
#Pump Simulation API
#Pump Simulation API
@app.get("/simulate/{task_id}/{pump}/{start_time}", 
        tags=["정수장"],
        summary="펌프 시뮬레이션 시작",
        response_model=OptimizeResponse 
        )
async def start_simulate(task_id: str, start_time: str, pump: str, background_tasks: BackgroundTasks):
    """
    24시간 펌프 시뮬레이션을 실행합니다.
    배수지별 예상 수위 변화량을 함께 계산합니다. 
    """

    background_tasks.add_task(pump_simulator, task_id, pump, start_time)
    return {"status" : "started", "task_id" : task_id, "type":"pump_optimizer"}

#Result View API
#Result View API
#Result View API
@app.get("/result/simulate/{task_id}", 
        tags=["정수장"],
        summary="펌프 운영 시뮬레이션 결과 조회",
        response_model=OptimizationResult
        )
async def get_result(task_id: str):
    """
    Redis에 저장된 펌프가동 시뮬레이션 결과 조회.
    """
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
        out["simulation_data"] = json.loads(raw["optimization_data"])
    if "start_from" in raw:
        out["start_from"] = raw["start_from"]
    if "error" in raw:
        out["error"] = raw["error"]
    return out