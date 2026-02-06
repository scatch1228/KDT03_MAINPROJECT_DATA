import torch.nn as nn

#==========Model Class===========
#==========Model Class===========
#==========Model Class===========
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