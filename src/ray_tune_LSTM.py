#(mypy) C:\WORKSPACE_MAINPROJECT\DATA_ANALYSIS\model>python -m i_resv.ray_tune.py
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
import ray
import joblib
import json
from flowpredictor import FlowPredictor

ray.init(address="auto", ignore_reinit_error=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tag={
    'j':10, #
    'e':28, #
    'd':33, #
    'a':40, ##
    'g':53, #
    'l':70 #
}

#data slicing
start = 1440 * 0
if (start != 0):
    start = start-2
end = start + 1439 * 365
split_rate = 0.7

#Sliding window config
window_size=180
forecast_size=15
lead_time=10 #forecast_size 보다 커지면 윈도우 사이에 갭이 생기므로 주의

#LSTM config
input_dim = 10
units = 64

#learning config
epochs=10
batch_size=64
lr=0.0001718750157412344 # < 0.08
dropout=0.5944708972513388
wd=1.37457e-05

def train_fn(config, train_dataset, test_dataset):
    units_ = config["units"]
    batch_size_ = config["batch_size"]
    lr_ = config["lr"]
    dropout_ = config["dropout"]
    weight_decay_ = config["weight_decay"]
    n_epochs = config["n_epochs"]

    # Create the DataLoaders
    train_loader_ = DataLoader(train_dataset, batch_size=batch_size_, shuffle=True) 
    test_loader_ = DataLoader(test_dataset, batch_size=batch_size_, shuffle=False)

    model_ = FlowPredictor(
        input_dim=input_dim, hidden_dim=units_, output_dim=forecast_size, dropout=dropout_
    ).to(device)
    opt_ = torch.optim.Adam(model_.parameters(), 
                        lr=lr_,
                        weight_decay=weight_decay_)
    mae_criterion = nn.L1Loss()

    for _ in range(n_epochs):
        model_.train()
        for bx, by in train_loader_:
            bx, by = bx.to(device), by.to(device)
            opt_.zero_grad()
            loss = mae_criterion(model_(bx), by.squeeze(-1))
            loss.backward()
            opt_.step()
        model_.eval()
        val_mae = 0.0
        with torch.no_grad():
            for bx, by in test_loader_:
                bx, by = bx.to(device), by.to(device)
                val_mae += mae_criterion(model_(bx), by.squeeze(-1)).item()
        val_mae /= len(test_loader_)
    tune.report({"val_mae": val_mae})

    del model_, opt_, train_loader_, test_loader_
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def tune_and_save(resv):

    #=======Call and Process Data======= 
    #=======Call and Process Data======= 
    #=======Call and Process Data======= 
    resv_flow = pd.read_csv(f"../data/rawdata/{tag[resv]}.csv")
    resv_flow.columns = ["id", "time", "resv_flow", "drop"]
    del resv_flow["id"]
    del resv_flow["drop"]

    resv_flow_temp = resv_flow[start:end].copy()
    resv_flow_temp.loc[:,'time'] = pd.to_datetime(resv_flow_temp['time'])
    resv_flow_temp.loc[:,'resv_flow'] = savgol_filter(resv_flow_temp['resv_flow'], window_length=31, polyorder=1)

    #======Call and Process Weather===== 
    #======Call and Process Weather===== 
    #======Call and Process Weather===== 
    filenames = [f'../data/weather/23{month:02d}.csv' for month in range(1, 13)]
    w_list = [pd.read_csv(f) for f in filenames]
    for w in w_list:
        w.columns = ['time','temperature','precipitate','humidity']
    weather = pd.concat(w_list, axis=0).reset_index(drop=True)

    time = pd.to_datetime(weather['time'])
    weather.loc[:,'time'] = time

    #결측치 처리
    weather['precipitate'] = weather['precipitate'].fillna(0)
    weather['temperature'] = weather['temperature'].interpolate(method='linear')
    weather['humidity'] = weather['humidity'].interpolate(method='linear')

    #======Merge Data=====
    #======Merge Data=====
    #======Merge Data=====
    df = pd.merge(resv_flow_temp, weather, how='inner', on='time')
    
    # Cyclical temporal features
    t = pd.to_datetime( df['time'])

    # 시간정보 (분 단위 하루 주기, T=1440)
    minute_of_day = t.dt.hour * 60 + t.dt.minute
    df['time_sin'] = 0.5 * np.sin(2 * np.pi * minute_of_day / 1440) + 0.5
    df['time_cos'] = 0.5 * np.cos(2 * np.pi * minute_of_day / 1440) + 0.5

    # 요일 (주간 주기, T=7)
    dow = t.dt.dayofweek
    df['dow_sin'] = 0.5 * np.sin(2 * np.pi * dow / 7) + 0.5
    df['dow_cos'] = 0.5 * np.cos(2 * np.pi * dow / 7) + 0.5

    # 계절 (연간 주기, T=365.25)
    doy = t.dt.dayofyear
    df['season_sin'] = 0.5 * np.sin(2 * np.pi * doy / 365.25) + 0.5
    df['season_cos'] = 0.5 * np.cos(2 * np.pi * doy / 365.25) + 0.5

    #=====Train/Test Split=====
    #=====Train/Test Split=====
    #=====Train/Test Split=====
    split_index = int(len(df) * split_rate)
    print(split_index)
    df_train = df.iloc[:split_index] 
    df_test = df.iloc[split_index:]

    #=====Normalization=====
    #=====Normalization=====
    #=====Normalization=====
    feature_cols = ['resv_flow', 'temperature', 'precipitate', 'humidity',
                    'time_sin', 'time_cos', 'dow_sin', 'dow_cos', 'season_sin', 'season_cos'
                    ]
    target_cols = ['resv_flow'] 

    scaler_x = MinMaxScaler(feature_range=(0,1))
    X_train_scaled = scaler_x.fit_transform(df_train[feature_cols])
    X_test_scaled = scaler_x.transform(df_test[feature_cols])

    scaler_y = MinMaxScaler(feature_range=(0,1))
    y_train_scaled = scaler_y.fit_transform(df_train[target_cols])
    y_test_scaled = scaler_y.transform(df_test[target_cols])

    #=====Create Sliding Window=====
    #=====Create Sliding Window=====
    #=====Create Sliding Window=====
    def create_sliding_windows(data, target, window_size, forecast_size, lead_time=0):
        stop_index = len(data) - window_size - forecast_size - lead_time
        X = [data[i:i+window_size] for i in range(stop_index)]
        y = [target[i+window_size+lead_time : i+window_size+forecast_size+lead_time] for i in range(stop_index)]
        
        return np.array(X), np.array(y)


    X_train, y_train = create_sliding_windows(data=X_train_scaled, 
                                target=y_train_scaled, 
                                window_size=window_size, 
                                forecast_size=forecast_size,
                                lead_time=lead_time)

    X_test, y_test = create_sliding_windows(data=X_test_scaled, 
                                target=y_test_scaled, 
                                window_size=window_size, 
                                forecast_size=forecast_size,
                                lead_time=lead_time)

    #=====Convert Train/Test into Tensor
    #=====Convert Train/Test into Tensor
    #=====Convert Train/Test into Tensor
    X_train, X_test, y_train, y_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test), torch.FloatTensor(y_train), torch.FloatTensor(y_test)
    # Wrap tensors into a Dataset object
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    trainable_with_data = tune.with_parameters(
        train_fn, 
        train_dataset=train_dataset, 
        test_dataset=test_dataset
    )

    param_space = {
        "units": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([32, 64, 128]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "dropout": tune.uniform(0.1, 0.5),
        "n_epochs": tune.choice([10, 15, 20]),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
    }

    def shortened_path(trial):
        return f"trial_{trial.trial_id}"


    tuner = Tuner(
        tune.with_resources(trainable_with_data, {"gpu": 1}),
        param_space=param_space,
        tune_config=TuneConfig(
            num_samples=10, 
            metric="val_mae", 
            mode="min", 
            max_concurrent_trials=1,
            trial_dirname_creator=shortened_path
        ),
        run_config=RunConfig(
            storage_path=r"C:\WORKSPACE_MAINPROJECT\DATA_ANALYSIS\ray_results",
            name=f"{resv}_resv_model_tuning"
        )
    )
    results = tuner.fit()
    best_result = results.get_best_result()
    best_config = best_result.config

    units = best_config["units"]
    batch_size = best_config["batch_size"]
    lr = best_config["lr"]
    dropout = best_config["dropout"]
    epochs = best_config["n_epochs"]
    wd = best_config["weight_decay"]

    print("Best config:", best_config)
    print("Best val_mae:", best_result.metrics.get("val_mae"))
    print("→ 위 설정으로 아래 최종 학습을 진행합니다.\n")

    # ---------------------------------------------------------------------------
    # 최적 파라미터로 데이터 재구성 후 최종 학습
    # ---------------------------------------------------------------------------

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = FlowPredictor(
        input_dim=input_dim, hidden_dim=units, output_dim=forecast_size, dropout=dropout
    ).to(device)
    mae_criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = wd)

    for epoch in range(epochs):
        model.train()
        runnintrain_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = mae_criterion(outputs, batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            runnintrain_loss += loss.item()
        avmae_train = runnintrain_loss / len(train_loader)

        model.eval()
        total_mae_val = 0.0
        all_preds = []
        all_tests = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                total_mae_val += mae_criterion(outputs, batch_y.squeeze(-1)).item()
                all_preds.append(outputs.cpu())
                all_tests.append(batch_y.cpu())
        average_mae_val = total_mae_val / len(test_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], MAE Loss: {avmae_train:.4f}, MAE Val Loss: {average_mae_val:.4f}"
            )

    y_pred_np = torch.cat(all_preds).numpy()
    y_test_np = torch.cat(all_tests).numpy()

    y_pred_original = scaler_y.inverse_transform(y_pred_np.reshape(-1, 1)).reshape(
        y_pred_np.shape
    )
    y_test_original = scaler_y.inverse_transform(y_test_np.reshape(-1, 1)).reshape(
        y_test_np.shape
    )


    #=====Data Plot=====
    #=====Data Plot=====
    #=====Data Plot=====
    sample_idx0 = 1440 * 0
    interval = 60*10 #Max = total interval (minute) * 0.2
    sample_idx1 = sample_idx0 + interval

    y_test_flattened = y_test_original[:,0].flatten()
    y_test_plot = y_test_flattened[sample_idx0 : sample_idx1]

    y_pred_flattened = y_pred_original[:,0].flatten()
    y_pred_plot = y_pred_flattened[sample_idx0 : sample_idx1]

    #Windows
    y_pred2_plot = y_pred_original[0]
    for i in range(interval//forecast_size):
        y_pred2_plot = np.concatenate((y_pred2_plot,y_pred_original[i*forecast_size]))

    plt.figure(figsize=(10, 5))

    plt.plot(range(len(y_test_plot)), y_test_plot, label='Actual Flow')#, marker='o')
    plt.plot(range(len(y_pred_plot)), y_pred_plot, label='Predicted Flow')#, marker='x')
    plt.plot(range(len(y_pred2_plot)), y_pred2_plot, label=f'{forecast_size}-min Pred Flow', color='green')


    plt.title(f"{interval}-Minute Forecast Reality Check")
    plt.xlabel("Minutes into Future")   
    plt.ylabel("Flow out (m^3/hour)")
    plt.legend()
    plt.grid(True)
    #plt.ylim(20, 100) 
    plt.show()

    #============Save Artifacts==============
    #============Save Artifacts==============
    #============Save Artifacts==============

    # 1. Save the PyTorch Model Weights
    # We move the model to CPU before saving to ensure it can be loaded even on servers without a GPU.
    model.cpu()
    torch.save(model.state_dict(), f'{resv}_resv_flow_model.pth')

    # 2. Save the Scaler
    joblib.dump(scaler_x, f'{resv}_resv_scaler_x.pkl')
    joblib.dump(scaler_y, f'{resv}_resv_scaler_y.pkl')

    # 3. Save the Hyperparams
    hyperparams = {
    "units": units,
    "forecast_size": forecast_size,
    "dropout": dropout
    }

    with open(f'{resv}_resv_config.json', 'w') as f:
        json.dump(hyperparams, f)

    print("Model and Scaler artifacts have been saved successfully.")

print('무슨 배수지?:')
tune_and_save(resv=input())
