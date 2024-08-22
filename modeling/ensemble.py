import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
import time
from tqdm import tqdm

# Define Network Architecture

class DeepNet(torch.nn.Module):

    def __init__(self, d_input, d_inner, d_out, n_layers):
        super(DeepNet, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.d_out = d_out

        self.input_layer = nn.Linear(self.d_input, self.d_inner)
        self.layers = nn.ModuleList([nn.Linear(self.d_inner, self.d_inner) for i in range(n_layers)])
        self.output_layer = nn.Linear(self.d_inner, self.d_out)

    def forward(self, x):
        x = self.input_layer(x)
        for i, l in enumerate(self.layers):
            x = self.layers[i // 2](x) + l(x)
        x = self.output_layer(x)
        return x

def forest_regression(x):
    return torch.tensor(list(range(min(300, x.shape[0])))).to(device)[0:256].unsqueeze(1)

class Ensemble(torch.nn.Module):

    def __init__(self, d_input, d_inner, d_out, n_layers):
        super(Ensemble, self).__init__()

        self.combiner_net = DeepNet(d_input, d_inner, d_out, n_layers)

    def forward(self, x):
        with torch.no_grad():
            y1 = deep_regression(x)
            y2 = forest_regression(x)
            x = torch.cat([y1, y2, x], dim=1)
        x = self.combiner_net(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: " + str(device))

df = pd.read_csv("training.csv")

columns_to_drop = ['standardized_operator_name', 'ffs_frac_type', 'relative_well_position',
                   'batch_frac_classification', 'well_family_relationship', 'frac_type']  # Specify columns to remove
df = df.drop(columns_to_drop, axis=1)  # axis=1 indicates columns

# replace NaNs
for col in df.columns:
    mode = df[col].mode().iloc[0]
    df[col] = df[col].fillna(mode)

X = df.drop(["OilPeakRate"], axis=1)
y = df["OilPeakRate"]

def min_max_scaling(df):
    return (df - df.min()) / (df.max() - df.min())

X = min_max_scaling(X.copy())

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        X = X.astype(float)  # Convert all columns to float (adjust if needed)
        y = y.astype(float)
        X_array = X.values
        y_array = y.values

        self.X = torch.from_numpy(X_array.astype(np.float32)).to(device)
        self.y = torch.from_numpy(y_array.astype(np.float32)).to(device)
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

# Model Data Parameters
batch_size = 64 * 4

# Instantiate training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)

train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break

# Regression Model
deep_regression = DeepNet(d_input=24, d_inner=100, d_out=1, n_layers=10).to(device)
deep_regression.load_state_dict(torch.load("models/cont_deep_regression.pth"))

# Tree Model
# TBD

# Ensemble
ensemble = Ensemble(d_input=2 + 24, d_inner=100, d_out=1, n_layers=10).to(device)

loss_fn = nn.MSELoss()

optimizer = optim.SGD(ensemble.parameters(), lr=0.005, momentum=0.0) # optim.RMSprop(model.parameters())

num_epochs = 30

min_loss = float('inf')

for epoch in tqdm(range(num_epochs)):
    avg_loss = 0
    cnt = 0
    for X2, y in train_dataloader:
        cnt += 1
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = ensemble(X2)
        loss = loss_fn(pred, y.unsqueeze(-1))
        avg_loss += math.sqrt(loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(ensemble.state_dict(), "ensemble.pth")
        loss.backward()
        nn.utils.clip_grad_norm_(ensemble.parameters(), 5)
        optimizer.step()
    
    print("\nEpoch (" + str(epoch) + "): " + str(avg_loss / cnt))

print("Training Complete")

avg_loss = 0
cnt = 0
for X2, y in test_dataloader:
    cnt += 1
    
    pred = ensemble(X2)
    loss = loss_fn(pred, y.unsqueeze(-1))
    print(loss.item())
    avg_loss += math.sqrt(loss.item())

print("\nTest: " + str(avg_loss / cnt))