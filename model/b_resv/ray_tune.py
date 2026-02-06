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

#if __name__ == "__main__":
#    if ray.is_initialized():
#        ray.shutdown()

ray.init(ignore_reinit_error=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

start = 1440 * 0
if (start != 0):
    start = start-2
end = start + 1439 * 365
split_rate = 0.7

window_size = 180
forecast_size = 15
lead_time = 1

units = 32
epochs = 15
batch_size = 128
lr=0.00280028
dropout=0.542298
wd=1.37457e-05

resv_flow = pd.read_csv("../../data/rawdata/46.csv")
resv_flow.columns = ["id", "time", "resv_flow", "drop"]
del resv_flow["id"]
del resv_flow["drop"]

resv_flow_temp = resv_flow[start:end].copy()
resv_flow_temp["time"] = pd.to_datetime(resv_flow_temp["time"])
resv_flow_temp["savgol_smooth"] = savgol_filter(resv_flow_temp["resv_flow"], window_length=31, polyorder=1)

scaler = MinMaxScaler(feature_range=(0, 1))
col = resv_flow_temp.columns[-1]
resv_flow_temp["normalized_flow"] = scaler.fit_transform(resv_flow_temp[[col]])
resv_flow_values = resv_flow_temp["normalized_flow"].values

def create_slidinwindows(data, window_size=60, lead_time=1, forecast_size=10):
    stop_index = len(data) - window_size - forecast_size - lead_time
    X = [data[i : i + window_size] for i in range(stop_index)]
    y = [
        data[i + window_size + lead_time : i + window_size + forecast_size + lead_time]
        for i in range(stop_index)
    ]
    return np.array(X), np.array(y)

# ---------------------------------------------------------------------------
# Ray Tune 사용을 위한 모델 클래스
# ---------------------------------------------------------------------------
class FlowPredictor(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout, input_dim=1):
        super(FlowPredictor, self).__init__()
        #Layer1
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        #Layer2
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        #Layer3
        self.lstm3 = nn.LSTM(hidden_dim // 2, hidden_dim // 4, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)

        #Last layer
        self.fc = nn.Linear(hidden_dim // 4, output_dim)
        #self.fc = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm1_out, _ = self.lstm1(x)
        out = self.dropout1(lstm1_out)

        lstm2_out, _ = self.lstm2(lstm1_out)
        out = self.dropout2(lstm2_out)

        lstm3_out, _ = self.lstm3(lstm2_out)

        # Use the last hidden state for prediction
        last_hidden = lstm3_out[:, -1, :]
        out = self.dropout3(last_hidden)
        #last_hidden = lstm2_out[:, -1, :]
        #out = self.dropout2(last_hidden)
        out = self.fc(out)
        return out

def train_fn(config):
    units_ = config["units"]
    batch_size_ = config["batch_size"]
    lr_ = config["lr"]
    dropout_ = config["dropout"]
    weight_decay_ = config["weight_decay"]
    n_epochs = config["n_epochs"]

    X_, y_ = create_slidinwindows(
        data=resv_flow_values, window_size=window_size, lead_time=lead_time, forecast_size=forecast_size
    )
    X_ = X_.reshape((X_.shape[0], X_.shape[1], 1))
    si = int(len(X_) * split_rate)
    X_tr, X_te = X_[:si], X_[si:]
    y_tr, y_te = y_[:si], y_[si:]

    X_tr = torch.FloatTensor(X_tr)
    X_te = torch.FloatTensor(X_te)
    y_tr = torch.FloatTensor(y_tr)
    y_te = torch.FloatTensor(y_te)
    train_loader_ = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size_, shuffle=True
    )
    test_loader_ = DataLoader(
        TensorDataset(X_te, y_te), batch_size=batch_size_, shuffle=False
    )

    model_ = FlowPredictor(
        input_dim=1, hidden_dim=units_, output_dim=forecast_size, dropout=dropout_
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
            loss = mae_criterion(model_(bx), by)
            loss.backward()
            opt_.step()
        model_.eval()
        val_mae = 0.0
        with torch.no_grad():
            for bx, by in test_loader_:
                bx, by = bx.to(device), by.to(device)
                val_mae += mae_criterion(model_(bx), by).item()
        val_mae /= len(test_loader_)
    tune.report({"val_mae": val_mae})

    del model_, opt_, train_loader_, test_loader_
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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
    tune.with_resources(train_fn, {"gpu": 1}),
    param_space=param_space,
    tune_config=TuneConfig(
        num_samples=10, metric="val_mae", mode="min", max_concurrent_trials=1,
        trial_dirname_creator=shortened_path
    ),
    run_config=RunConfig(
        storage_path=r"C:\WORKSPACE_MAINPROJECT\DATA_ANALYSIS\ray_results",
        name="resv_model_tuning"
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

print("Best config:", best_config)
print("Best val_mae:", best_result.metrics.get("val_mae"))
print("→ 위 설정으로 아래 최종 학습을 진행합니다.\n")
# ---------------------------------------------------------------------------
# 최적 파라미터로 데이터 재구성 후 최종 학습
# ---------------------------------------------------------------------------

X, y = create_slidinwindows(
    data=resv_flow_values,
    window_size=window_size,
    lead_time=lead_time,
    forecast_size=forecast_size,
)
X = X.reshape((X.shape[0], X.shape[1], 1))

split_index = int(len(X) * split_rate)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = FlowPredictor(
    input_dim=1, hidden_dim=units, output_dim=forecast_size, dropout=dropout
).to(device)
mae_criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    runnintrain_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = mae_criterion(outputs, batch_y)
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
            total_mae_val += mae_criterion(outputs, batch_y).item()
            all_preds.append(outputs.cpu())
            all_tests.append(batch_y.cpu())
    average_mae_val = total_mae_val / len(test_loader)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], MAE Loss: {avmae_train:.4f}, MAE Val Loss: {average_mae_val:.4f}"
        )

y_pred_np = torch.cat(all_preds).numpy()
y_test_np = torch.cat(all_tests).numpy()

y_pred_original = scaler.inverse_transform(y_pred_np.reshape(-1, 1)).reshape(
    y_pred_np.shape
)
y_test_original = scaler.inverse_transform(y_test_np.reshape(-1, 1)).reshape(
    y_test_np.shape
)

sample_idx0 = 1440 * 0
interval = 60*24 #Max = total interval (minute) * 0.2
sample_idx1 = sample_idx0 + interval

y_test_flattened = y_test_original[:,0].flatten()
y_test_plot = y_test_flattened[sample_idx0 : sample_idx1]

y_pred_flattened = y_pred_original[:,5].flatten() #인덱스 적절하게 조정하기
y_pred_plot = y_pred_flattened[sample_idx0 : sample_idx1]

#Windows
y_pred2_plot = y_pred_original[0]
first_n_days = 1440 * 1
for i in range(first_n_days//forecast_size):
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
#plt.ylim(20, 120) 
plt.show()

#============Save Artifacts==============
#============Save Artifacts==============
#============Save Artifacts==============

# 1. Save the PyTorch Model Weights
# We move the model to CPU before saving to ensure it can be loaded even on servers without a GPU.
model.cpu()
torch.save(model.state_dict(), 'b_resv_flow_model.pth')

# 2. Save the Scaler
joblib.dump(scaler, 'b_resv_scaler.pkl')

# 3. Save the Hyperparams
hyperparams = {
"units": units,
"forecast_size": forecast_size,
"dropout": dropout
}

with open('b_resv_config.json', 'w') as f:
    json.dump(hyperparams, f)

print("Model and Scaler artifacts have been saved successfully.")
