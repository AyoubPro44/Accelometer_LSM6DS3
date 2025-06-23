import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load and clean data
df = pd.read_csv('motion_dataset.csv', names=["x", "y", "z", "state"])
df = df[pd.to_numeric(df["x"], errors="coerce").notnull()]
df = df[pd.to_numeric(df["y"], errors="coerce").notnull()]
df = df[pd.to_numeric(df["z"], errors="coerce").notnull()]

df["x"] = df["x"].astype(float)
df["y"] = df["y"].astype(float)
df["z"] = df["z"].astype(float)

x = df[["x", "y", "z"]].values
y = df["state"].values

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(np.unique(y_enc))

# One-hot encoding manually for PyTorch
y_onehot = np.eye(num_classes)[y_enc]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloaders
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_size = int(len(train_dataset) * 0.9)
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=32)


# Define the model
class MotionClassifier(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(MotionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


model = MotionClassifier(3, 64, 32, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, torch.argmax(batch_y, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            outputs = model(val_x)
            _, predicted = torch.max(outputs.data, 1)
            total += val_y.size(0)
            correct += (predicted == torch.argmax(val_y, dim=1)).sum().item()

    val_accuracy = correct / total
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Test evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for test_x, test_y in test_loader:
        outputs = model(test_x)
        _, predicted = torch.max(outputs.data, 1)
        total += test_y.size(0)
        correct += (predicted == torch.argmax(test_y, dim=1)).sum().item()

test_accuracy = correct / total
print(f"Test accuracy: {test_accuracy:.2f}")

# Save model
torch.save(model.state_dict(), 'motion_classifier3.pth')