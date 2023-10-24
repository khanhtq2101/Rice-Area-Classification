import torch
import torch.nn as nn

class Classifier1D(nn.Module):
  def __init__(self, polarlization = 'both'):
    super().__init__()
    self.trained_epochs = 0

    self.polarlization = polarlization
    if polarlization == 'both':
      n_chanels = 2
    else:
      n_chanels = 1

    self.cnn1d = nn.Sequential(
        nn.Conv1d(in_channels= n_chanels, out_channels= 4, kernel_size=3),
        nn.LeakyReLU(0.1), 

        nn.Conv1d(in_channels= 4, out_channels= 8, kernel_size= 3),
        nn.LeakyReLU(0.1),

        nn.Conv1d(in_channels= 8, out_channels= 8, kernel_size= 3),
        nn.LeakyReLU(0.1),

        nn.Conv1d(in_channels= 8, out_channels= 10, kernel_size= 3),
        nn.LeakyReLU(0.1)
    )

    self.flatten = nn.Flatten(start_dim=1, end_dim= -1)

    self.fc = nn.Sequential(
        nn.Linear(220, 64),
        nn.LeakyReLU(0.1),
        nn.Linear(64, 16),
        nn.LeakyReLU(0.1),
        nn.Linear(16, 5),
    )
  
  def forward(self, x):
    x = self.cnn1d(x)
    x = self.flatten(x)
    x = self.fc(x)
    return x
  
class RNNClassifier(nn.Module):
  def __init__(self, n_layers = 2):
    super().__init__()
    self.trained_epochs = 0
    self.n_layers = n_layers

    self.rnn = nn.RNN(input_size= 2, hidden_size= 60, num_layers= n_layers, batch_first= True)
    self.fc = nn.Sequential(
        nn.Linear(60, 32),
        nn.LeakyReLU(0.1),
        nn.Linear(32, 5))
  
  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = self.rnn(x)
    x = x[1][-1, :, :]
    x = self.fc(x)
    return x