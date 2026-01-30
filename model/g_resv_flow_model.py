import numpy as np
import json
import torch
import scipy
import torch.nn as nn  
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

#==========Version Check===========
#==========Version Check===========
#==========Version Check===========
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"torch: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def g_resv_flow_pred_json():
    #==========Params===========
    #==========Params===========
    #==========Params===========
    #data slicing
    start = 1440 * 1 # +1일차
    if (start != 0):
        start = start-2
    end = start + 1440 * 5 #일간
    split_rate = 1 - 1/5 

    #Sliding window config
    window_size=120
    forecast_size=30
    lead_time=1 #forecast_size 보다 커지면 윈도우 사이에 갭이 생기므로 주의

    #LSTM config
    units = 64

    #learning config
    epochs=10
    batch_size=64
    lr=0.01 # < 0.08
    dropout=0.3

    #==========Data Load===========
    #==========Data Load===========
    #==========Data Load===========
    g_resv_flow = pd.read_csv('../data/rawdata/53.csv')
    g_resv_flow.columns = ['id', 'time', 'g_resv_flow', 'drop']
    del g_resv_flow['id']
    del g_resv_flow['drop']

    g_resv_flow_temp = g_resv_flow[start:end]

    time = g_resv_flow_temp['time']
    time = pd.to_datetime(time)
    g_resv_flow_temp['time'] = time

    #==========Preprocessing & Normalization===========
    #==========Preprocessing & Normalization===========
    #==========Preprocessing & Normalization===========
    g_resv_flow_temp['savgol_smooth'] = savgol_filter(g_resv_flow_temp['g_resv_flow'], window_length=31, polyorder=1)

    scaler = MinMaxScaler(feature_range=(0,1))
    column_to_normalize = g_resv_flow_temp.columns[-1]
    g_resv_flow_temp['normalized_flow'] = scaler.fit_transform(g_resv_flow_temp[[column_to_normalize]])

    #==========Create Sliding Windows===========
    #==========Create Sliding Windows===========
    #==========Create Sliding Windows===========
    def create_sliding_windows(data, window_size=60, lead_time=1, forecast_size=10):
        #기존 분단위 슬라이딩 코드
        stop_index = len(data) - window_size - forecast_size# -lead_time
        X = [data[i:i+window_size] for i in range(stop_index)]
        y = [data[i+window_size : i+window_size+forecast_size] for i in range(stop_index)]
        
        #신규 lead_time분단위 슬라이딩 코드: 윈도우를 1분씩 뒤로 미는 것이 아니라 x분씩 뒤로 민다. 과도한 학습을 방지하기 위해 작성해보았음.
        #stop_index = int((len(data) - window_size - forecast_size)/lead_time)
        #X = [data[i*lead_time : i*lead_time+window_size] for i in range(stop_index)]
        #y = [data[i*lead_time+window_size : i*lead_time+window_size+forecast_size] for i in range(stop_index)]

        return np.array(X), np.array(y)

    data = g_resv_flow_temp['normalized_flow'].values
    X,y = create_sliding_windows(data=data, window_size=window_size, lead_time=lead_time, forecast_size=forecast_size)
    X = X.reshape((X.shape[0], X.shape[1],1))

    #==========Train Test Split===========
    #==========Train Test Split===========
    #==========Train Test Split===========
    split_index = int(len(X) * split_rate)
    print(split_index)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    #==========Conver it to Tensor===========
    #==========Conver it to Tensor===========
    #==========Conver it to Tensor===========
    X_train, X_test, y_train, y_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test), torch.FloatTensor(y_train), torch.FloatTensor(y_test)

    #==========Model Class===========
    #==========Model Class===========
    #==========Model Class===========
    class FlowPredictor(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=units, output_dim=forecast_size, dropout=dropout):
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
            out = self.fc(out)
            return out
        
    #==========Execute Learning===========
    #==========Execute Learning===========
    #==========Execute Learning===========
    #Univariate LSTM with MSE Loss
    #Move the model and train/test data to CUDA
    model = FlowPredictor().to(device)

    # Wrap tensors into a Dataset object
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #mse_criterion = nn.MSELoss() # For MSE
    mae_criterion = nn.L1Loss() # Equivalent to mean_absolute_error
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #Learning
    for epoch in range(epochs):
        model.train()

        running_train_loss = 0.0
        #Train
        for batch_X, batch_y in train_loader:
            # Move batch to CUDA
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (Batch level)
            outputs = model(batch_X)
            loss = mae_criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        # Calculate average training loss for this epoch
        avg_mae_train = running_train_loss / len(train_loader)
        
        # Validation
        all_preds = []
        all_tests = []

        model.eval()
        total_mae_val = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                # Move ONLY the small batch to the GPU
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)

                loss = mae_criterion(outputs, batch_y)
                total_mae_val += loss.item()

                # Move back to CPU and convert to list/numpy
                all_preds.append(outputs.cpu())
                all_tests.append(batch_y.cpu())

        average_mae_val = total_mae_val / len(test_loader)
        if(epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], MAE Loss: {avg_mae_train:.4f}, MAE Val Loss: {average_mae_val:.4f}')

    # Concatenate all batches into a single numpy array
    y_pred_np = torch.cat(all_preds).numpy()
    y_test_np = torch.cat(all_tests).numpy()


    #==========Denormalize===========
    #==========Denormalize===========
    #==========Denormalize===========
    y_pred_original = scaler.inverse_transform(y_pred_np.reshape(-1,1)).reshape(y_pred_np.shape)
    y_test_original = scaler.inverse_transform(y_test_np.reshape(-1,1)).reshape(y_test_np.shape)


    #==========Data Export===========
    #==========Data Export===========
    #==========Data Export===========
    y_pred_flattened = y_pred_original[:,0].flatten()
    y_pred_export = np.concatenate( (y_pred_flattened, y_pred_original[-1]))

    g_resv_flow_export = g_resv_flow_temp[-1440:]

    y_accuracy = (1-np.abs(y_pred_original - y_test_original) / (y_test_original+1))
    y_accuracy_export = np.concatenate((y_accuracy[:,0], y_accuracy[-1]))

    del g_resv_flow_export['savgol_smooth']
    del g_resv_flow_export['normalized_flow']
    del g_resv_flow_export['g_resv_flow']
    g_resv_flow_export['g_resv_flow_pred'] = y_pred_export
    g_resv_flow_export['accuracy'] = y_accuracy_export


    # 1. Convert the DataFrame rows to a list of dictionaries
    json_data = g_resv_flow_export.to_json(orient='records', date_format='iso')

    # 2. Parse that string back to a Python object and wrap it
    # (This step is necessary if you want the 'predictions' header)
    final_dict = {
        "g_resv_flow_pred": json.loads(json_data)
    }

    # 3. Convert the final dictionary to a JSON string for Redis
    json_for_redis = json.dumps(final_dict)
    
    return json_for_redis