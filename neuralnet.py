from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def plot_losses(loss):
  sns.lineplot(x=list(range(len(loss))), y=loss)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Loss during training")
  plt.show()

def plot_accuracies(accuracy):
  sns.lineplot(x=list(range(len(accuracy))), y=accuracy)
  plt.xlabel("Epoch")
  plt.ylabel("Test Accuracy")
  plt.title("Test accuracy during training")
  plt.show()

dataset = pd.read_json(r'data/word2vec_data.json', lines=True)

x_train, x_test, y_train, y_test = train_test_split(dataset['word2vec'], dataset['category'], test_size=0.2, random_state=42)

x_train = [x for x in x_train]
x_test = [x for x in x_test]

y_train = [x for x in y_train]
y_test = [x for x in y_test]
  
label_encoder = OneHotEncoder()
y_train_encoded = label_encoder.fit_transform(np.array(y_train).reshape(-1, 1)).todense()
y_test_encoded = label_encoder.fit_transform(np.array(y_test).reshape(-1, 1)).todense()

import torch
import torch.nn as nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(128, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 15),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.stack(x)
        return x

model = NeuralNetwork().to(device)

x_train = torch.Tensor(x_train).to(device)
y_train = torch.Tensor(y_train_encoded).to(device)

x_test = torch.Tensor(x_test).to(device)
y_test = torch.Tensor(y_test_encoded).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

losses = []
accuracies = []

for epoch in range(1000):
    optimizer.zero_grad()
    
    preds = model(x_train)
    loss = loss_fn(preds, y_train)

    loss.backward()
    losses.append(loss.item())
    optimizer.step()

    accuracy_preds = model(x_test)
    accuracy_preds = torch.argmax(accuracy_preds, dim=1)

    accuracy = torch.sum(accuracy_preds == torch.argmax(y_test, dim=1)) / accuracy_preds.shape[0]
    accuracies.append(accuracy.item())
    print("Epoch " + str(epoch) + ":", "Loss: " + str(loss.item()), "Test accuracy: " + str(accuracy.item()))

plot_losses(losses)
plot_accuracies(accuracies)