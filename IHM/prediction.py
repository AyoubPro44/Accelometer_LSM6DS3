import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Define the same model architecture
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

df = pd.read_csv("motion_state.csv", names=["x", "y", "z", "state"], dtype=str)
df = df[pd.to_numeric(df["x"], errors="coerce").notnull()]
df = df[pd.to_numeric(df["y"], errors="coerce").notnull()]
df = df[pd.to_numeric(df["z"], errors="coerce").notnull()]
df["state"] = df["state"].astype(str)  # Ensure it's string

le = LabelEncoder()
le.fit(df["state"])
num_classes = len(le.classes_)  # Should be 3 (or whatever the trained model had)

# Initialize and load trained model
model = MotionClassifier(3, 64, 32, num_classes)
model.load_state_dict(torch.load("motion_classifier.pth"))
model.eval()

# Example input (replace with real sensor data)
def predict_motion(x, y, z):
    input_tensor = torch.tensor([[x, y, z]], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = le.inverse_transform([predicted_class])[0]
    return predicted_label
