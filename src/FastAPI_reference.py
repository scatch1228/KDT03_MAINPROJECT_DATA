import asyncio
import json
import socket
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Path, HTTPException
from redis.asyncio import Redis

from src.g_resv_flow_model import g_resv_flow_pred_json

app = FastAPI()

SUZY_RESV_FLOW = 10  # 배수지 예측 모델 식별자
RESULT_EXPIRE_SECONDS = 3600  # 결과 Redis TTL (1시간)


# 1. Redis 연결 확인 후에 없는 경우
def is_redis_available(host: str, port: int, timeout: int = 1) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


redis_client = None
_redis_host = "10.125.121.184"
_redis_port = 6379

# 2. Redis 클라이언트 초기화
if is_redis_available(_redis_host, _redis_port):
    redis_client = Redis(host=_redis_host, port=_redis_port, decode_responses=True)
else:
    print(f"[WARNING] Redis 서버에 연결할 수 없습니다: {_redis_host}:{_redis_port}")


# 3. 결과 저장 함수
async def _save_result(
    task_id: str,
    status: str,
    prediction_data: str | None = None,
    predict_time: str | None = None,
    error: str | None = None,
):
    if redis_client is None:
        print(f"[{task_id}] Redis 미연결로 결과 저장 불가 (status={status})")
        return
    mapping = {"status": status}
    if prediction_data is not None:
        mapping["prediction_data"] = prediction_data
    if predict_time is not None:
        mapping["predict_time"] = predict_time
    if error is not None:
        mapping["error"] = error
    key = f"result:{task_id}"
    await redis_client.hset(key, mapping=mapping)
    await redis_client.expire(key, RESULT_EXPIRE_SECONDS)


# 4. 비동기 학습/예측 처리 함수
# 향후 모델로 변경해야 함
async def learning_process(task_id: str, suzy: int):
    print(f"[{task_id}] 학습 시작... (배수지: {suzy})")

    if redis_client is None:
        print(f"[{task_id}] Redis 미연결로 작업 중단")
        await _save_result(task_id, "error", error="Redis 서버 연결 불가")
        return

    if suzy != SUZY_RESV_FLOW:
        print(f"[{task_id}] unknown Suzy: {suzy}")
        await _save_result(task_id, "failed", error=f"unknown Suzy: {suzy}")
        return

    try:
        data = await asyncio.to_thread(g_resv_flow_pred_json)
        await _save_result(
            task_id, "completed", prediction_data=data[0], predict_time=data[1]
        )
        print(f"[{task_id}] Hash 데이터 저장 완료!")
    except Exception as e:
        print(f"[{task_id}] 오류: {e}")
        await _save_result(task_id, "error", error=str(e))


@app.get("/predict/{suzy}/{task_id}")
async def start_predict(
    background_tasks: BackgroundTasks,
    task_id: str = Path(..., min_length=1, description="태스크 ID"),
    suzy: int = Path(..., description="배수지 모델 식별자 (10=예측)"),
):
    background_tasks.add_task(learning_process, task_id, suzy)
    return {"status": "started", "task_id": task_id, "bae_suzy": suzy}


@app.get("/result/{task_id}")
async def get_result(task_id: str = Path(..., min_length=1, description="태스크 ID")):
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
    if "predict_time" in raw:
        out["predict_time"] = json.loads(raw["predict_time"])
    if "error" in raw:
        out["error"] = raw["error"]
    return out


# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
