import torch
from torch import nn
from functools import reduce



class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, bidirectional, num_cls, batch_first , dropout):
      super().__init__()
      self.rnn = nn.LSTM(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          bidirectional=bidirectional,
          batch_first=batch_first ,
          dropout=dropout
      )
      self.fc = nn.LazyLinear(num_cls)

  def forward(self, x):
      outputs, (hn, cn) = self.rnn(x)  # خروجی‌ها و hidden stateها
      # استفاده از آخرین hidden state (آخرین گام زمانی)
      # last_output = outputs[:, -1, :]  # شکل: (batch_size, hidden_size)
      # y = self.fc(last_output)
      # y = torch.sigmoid(y)
      # return y#.mean(dim=1)
      return outputs
  






class CNNLSTM(nn.Module):
    def __init__(self, input_size, cnn_hidden_size, rnn_hidden_size, num_layers, num_cls):
        super().__init__()

        # لایههای کانولوشن با پدینگ مناسب
        self.conv1 = nn.Conv1d(input_size, cnn_hidden_size, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_hidden_size)

        self.conv2 = nn.Conv1d(cnn_hidden_size, cnn_hidden_size, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_hidden_size)

        self.conv3 = nn.Conv1d(cnn_hidden_size, cnn_hidden_size, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(cnn_hidden_size)

        # غیرفعال کردن پولینگ
        # self.pool = nn.MaxPool1d(3, 3)

        # لایههای LSTM و Fully Connected
        self.lstm = nn.LSTM(cnn_hidden_size, rnn_hidden_size, num_layers, batch_first=True)
        self.fc = nn.LazyLinear(num_cls)

    def forward(self, x):
        # تغییر ابعاد برای کانولوشن
        x = x.permute(0, 2, 1)  # [batch, features, sequence]

        # عبور از لایههای کانولوشن (بدون کاهش طول دنباله)
        y = self.bn1(self.conv1(x)).relu()
        y = self.bn2(self.conv2(y)).relu()
        y = self.bn3(self.conv3(y)).relu()

        # غیرفعال کردن پولینگ
        # y = self.pool(y)

        # تغییر ابعاد برای LSTM
        y = y.permute(0, 2, 1)  # [batch, sequence, features]
        y, _ = self.lstm(y)

        # پیشبینی نهایی
        # y = self.fc(y[:, -1, :])
        # y = torch.sigmoid(y)
        return y
    



class LSTM_final_model(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, bidirectional, num_cls, batch_first , dropout):
      super().__init__()
      self.rnn = nn.LSTM(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          bidirectional=bidirectional,
          batch_first=batch_first ,
          dropout=dropout
      )
      self.fc = nn.LazyLinear(num_cls)

  def forward(self, x):
      outputs, (hn, cn) = self.rnn(x)  # خروجی‌ها و hidden stateها
      # استفاده از آخرین hidden state (آخرین گام زمانی)
      last_output = outputs[:, -1, :]  # شکل: (batch_size, hidden_size)
      y = self.fc(last_output)
      y = torch.sigmoid(y)
      return y




















