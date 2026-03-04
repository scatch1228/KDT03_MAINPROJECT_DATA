"""모델 및 설정을 로드하여 실제 추론 및 시뮬레이션 데이터를 생성하는 모듈."""

import json
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUZY_TO_LETTER = {4: "a", 7: "d", 8: "e", 10: "g", 13: "j", 15: "l"}
LETTER_TO_SUZY = {v: k for k, v in SUZY_TO_LETTER.items()}
RESERVOIR_LABELS = {
    "a": "a배수지",
    "d": "d배수지",
    "e": "e배수지",
    "g": "g배수지",
    "j": "j배수지",
    "l": "l배수지",
}

# 라이브러리 경로 설정
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from inference import ReservoirInferenceService


def _model_dir():
    return PROJECT_ROOT / "model"


def _config_path(letter):
    return _model_dir() / f"{letter}_resv_config.json"


def get_available_reservoirs():
    """설정 파일이 존재하는 모든 배수지 목록을 반환합니다."""
    return [(suzy_id, letter) for suzy_id, letter in SUZY_TO_LETTER.items()]


def load_config(letter):
    """배수지별 JSON 설정 파일을 로드합니다."""
    with open(_config_path(letter), encoding="utf-8") as f:
        return json.load(f)


def get_forecast_size(letter=None):
    """설정된 예측 구간 크기를 반환합니다."""
    target_letter = letter or "a"
    cfg = load_config(target_letter)
    return int(cfg["forecast_size"])


def get_reservoir_names_from_project():
    """배수지 ID와 이름 매핑 딕셔너리를 반환합니다."""
    return {sid: RESERVOIR_LABELS[let] for sid, let in SUZY_TO_LETTER.items()}


def get_facility_ids_from_project():
    """최적화 대상 시설 ID 목록을 반환합니다."""
    return sorted(list(SUZY_TO_LETTER.keys()))


def get_sample_prediction_from_project(suzy_id=None):
    """모델 설정을 반영한 합성 수요예측 데이터를 생성합니다."""
    letter = SUZY_TO_LETTER.get(suzy_id, "a")
    forecast_size = get_forecast_size(letter)
    total_len = (60 // forecast_size) * forecast_size

    t = np.linspace(0, 4 * np.pi, total_len)
    prediction_data = (
        (22 + 3 * np.sin(t) + np.random.RandomState(42).randn(total_len) * 0.5)
        .clip(15, 35)
        .tolist()
    )

    return {
        "task_id": "demo-from-model",
        "status": "completed",
        "prediction_data": prediction_data,
        "predict_from": "2024-01-15 14:00:00",
        "accuracy": str(round(0.85 + np.random.RandomState(7).rand() * 0.1, 2)),
    }


def get_sample_optimization_from_project():
    """배수지별 펌프 가동 및 수위 변동 시뮬레이션 데이터를 생성합니다."""
    facility_ids = get_facility_ids_from_project()
    timestamps = [f"2024-01-15T{i:02d}:00:00" for i in range(24)]

    results = []
    level_base = {fid: 2.5 for fid in facility_ids}
    for i, ts in enumerate(timestamps):
        h = i % 24
        pumps, price = (
            (2, 70.0)
            if (23 <= h or h < 9)
            else ((1, 200.0) if h in (10, 11, 17, 18, 19, 22) else (2, 130.0))
        )
        sim_levels = {
            str(fid): round(level_base[fid] + (pumps - 1.5) * 0.08, 2)
            for fid in facility_ids
        }
        results.append(
            {
                "timestamp": ts,
                "active_pumps": pumps,
                "sim_levels": sim_levels,
                "sim_cost": round((pumps * 150 / 60) * (price / 100), 2),
                "spill_m3_per_min": 0.0,
            }
        )
    return results


def _resv_configs_for_inference():
    """model/ 디렉터리 기준 pkl·pth 경로로 ReservoirInferenceService용 config를 만듭니다."""
    root = _model_dir()
    return {
        suzy_id: {
            "weights": str(root / f"{letter}_resv_flow_model.pth"),
            "scaler_x": str(root / f"{letter}_resv_scaler_x.pkl"),
            "scaler_y": str(root / f"{letter}_resv_scaler_y.pkl"),
            "config": str(root / f"{letter}_resv_config.json"),
        }
        for suzy_id, letter in SUZY_TO_LETTER.items()
        if (root / f"{letter}_resv_flow_model.pth").exists()
    }


def _demo_input_window(suzy_id, start_time):
    """추론용 225행 입력 DataFrame을 생성(generator 입력 형식과 동일한 형태)"""
    n_rows = 240
    base = pd.Timestamp(start_time) - pd.Timedelta(minutes=180)
    current_dir = Path(__file__).resolve().parent

    resv_path = current_dir / f"{suzy_id}.csv"
    weather_path = current_dir / "2401.csv"

    resv = pd.read_csv(resv_path)
    resv.columns = ['drop','collected_at', 'resv_flow', 'drop2']
    resv['collected_at'] = pd.to_datetime(resv['collected_at'])

    weather = pd.read_csv(weather_path)
    weather.columns = ['collected_at','temperature','precipitate','humidity']
    weather['collected_at'] = pd.to_datetime(weather['collected_at'])

    df_merged = pd.merge(resv, weather, on='collected_at', how='inner').drop(columns=['drop', 'drop2'])
    df = df_merged[df_merged['collected_at'] >= base].head(n_rows).reset_index(drop=True)


    if int(suzy_id) in [4,7]:
        # A배수지: 9개 피처
        t = df['collected_at']
        minute_of_day = t.dt.hour * 60 + t.dt.minute
        time_sin = 0.5 * np.sin(2 * np.pi * minute_of_day / 1440) + 0.5
        time_cos = 0.5 * np.cos(2 * np.pi * minute_of_day / 1440) + 0.5

        dow = t.dt.dayofweek
        dow_sin = 0.5 * np.sin(2 * np.pi * dow / 7) + 0.5
        dow_cos = 0.5 * np.cos(2 * np.pi * dow / 7) + 0.5

        doy = t.dt.dayofyear
        season_sin = 0.5 * np.sin(2 * np.pi * doy / 365.25) + 0.5
        season_cos = 0.5 * np.cos(2 * np.pi * doy / 365.25) + 0.5

        df['time_sin']=time_sin
        df['time_cos']=time_cos
        df['dow_sin']=dow_sin
        df['dow_cos']=dow_cos
        df['season_sin']=season_sin
        df['season_cos']=season_cos
        
        train_df = df[:-15] # 길이 225
        val_df = df['resv_flow'][-60:].values
        columns = ['resv_flow', 'temperature', 'humidity',
                'time_sin', 'time_cos', 'dow_sin', 
                'dow_cos', 'season_sin', 'season_cos'
                ]
    else:
        train_df = df[:-15] # 길이 225
        val_df = df['resv_flow'][-60:].values
        columns = ['resv_flow', 'temperature', 'precipitate','humidity',]
    
    return train_df[columns], val_df


def _predict_with_model_only(suzy_id, start_time):
    """model/ 의 pkl·pth만 사용해 수요 예측 """
    window_size = 180
    forecast_size = 15
    total_forecast_size = 60
    input_dim = 9

    configs = _resv_configs_for_inference()

    service = ReservoirInferenceService(
        configs, input_dim=input_dim, window_size=window_size
    )
    input_window, val_df = _demo_input_window(suzy_id, start_time)

    prediction = service.predict(suzy_id, input_window[:window_size].copy())
    for i in range(1, total_forecast_size // forecast_size):
        start_idx = i * forecast_size
        end_idx = window_size + i * forecast_size
        pred_next = service.predict(suzy_id, input_window[start_idx:end_idx].copy())
        prediction = np.concatenate((prediction, pred_next))

    prediction = prediction.flatten()

    accuracy = np.mean( (1- np.abs(val_df - prediction) / (val_df+0.01)) )
    json_pred = json.dumps(prediction.tolist())
    json_val = json.dumps(val_df.tolist())
    json_date = str(pd.Timestamp(start_time))
    json_acc = str(accuracy)
    return json_pred, json_val, json_date, json_acc


def try_run_real_inference(suzy_id, start_time):
    """수요예측: model/ 의 pkl·pth만 사용해 추론"""
    orig_cwd = os.getcwd()
    os.chdir(SRC_PATH)
    try:
        json_pred, json_val, json_date, json_acc = _predict_with_model_only(suzy_id, start_time)
    finally:
        os.chdir(orig_cwd)

    return {
        "task_id": "demo-real",
        "status": "completed",
        "prediction_data": json.loads(json_pred),
        "actual_data": json.loads(json_val),
        "predict_from": json_date,
        "accuracy": json_acc,
    }


def try_run_real_optimization(start_time):
    """펌프 최적화: 샘플 최적화 데이터를 반환"""
    return get_sample_optimization_from_project()
