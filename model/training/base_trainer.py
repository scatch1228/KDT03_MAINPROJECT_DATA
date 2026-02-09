#(mypy) C:\WORKSPACE_MAINPROJECT\DATA_ANALYSIS\model>python -m g_resv.ray_tune.py
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from ray import tune, train
from ray.tune import Tuner, TuneConfig, RunConfig
from flowpredictor import FlowPredictor
import ray

class ReservoirTrainer:
    def __init__(self, reservoir_id: str, config: dict):
        self.reservoir_id = reservoir_id
        self.config = config

    def train(self, data_path: str):
        # 공통 학습 로직
        pass

    def tune_hyperparameters(self):
        # Ray Tune 로직
        pass

    def save_artifacts(self, output_dir: str):
        # 모델, 스케일러, 설정 저장
        pass

# 사용 예시
trainer = ReservoirTrainer('g_resv', config)
trainer.train('../data/rawdata/53.csv')

