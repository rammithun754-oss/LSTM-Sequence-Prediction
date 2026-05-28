import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --------------------------------
# Generate Sequential Dataset
# --------------------------------
sequence_data = []
targets = []
for i in range(1, 101):   # 100 samples
    seq = [i, i+1, i+2, i+3]
    sequence_data.append(seq)
    targets.append(i+4)

# Convert to tensors
X = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

# Normalize (helps training stability)
X = X / 100.0
y = y / 100.0

# --------------------------------
# LSTM Model
# --------------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        last_output = output[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

# Initialize model
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --------------------------------
# Training
# --------------------------------
epochs = 1000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Plot training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# --------------------------------
# Testing
# --------------------------------
test_sequence = torch.tensor([[[15],[16],[17],[18]]], dtype=torch.float32) / 100.0
prediction = model(test_sequence)

# Rescale back
predicted_value = prediction.item() * 100
print("\nPredicted Next Value:")
print(predicted_value)
