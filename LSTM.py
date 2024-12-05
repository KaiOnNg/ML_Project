from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import torch.nn as nn

# Diagnostic info
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name()}")

# Feature Construction
def create_features(df):
    """Create linear and non-linear features."""
    df_new = df.copy()
    numeric_features = df.select_dtypes(include=[np.number]).columns

    # Square and cube terms
    for feature in numeric_features:
        df_new[f"{feature}_squared"] = df[feature] ** 2
        df_new[f"{feature}_cubed"] = df[feature] ** 3

    # Log terms
    for feature in numeric_features:
        if (df[feature] > 0).all():
            df_new[f"{feature}_log"] = np.log(df[feature] + 1)

    # Interaction terms
    important_features = ["Adj Close", "Volume", "RSI"]
    for i in range(len(important_features)):
        for j in range(i + 1, len(important_features)):
            df_new[f"{important_features[i]}_{important_features[j]}_interaction"] = (
                df[important_features[i]] * df[important_features[j]]
            )
    return df_new

# Create sequences for LSTM
def create_sequences(data, target, sequence_length=30):
    """Create sequences of past data for LSTM."""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load and preprocess data
df = pd.read_csv('SP500_with_indicators_^GSPC.csv').dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.drop(columns=['Close'])

# Feature construction
df = create_features(df)

# Use the features and target
features = df.drop(columns=["Adj Close"]).columns
target = df["Adj Close"]

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Split into train-validation-test sets
split_1 = int(len(df) * 0.7)
split_2 = int(len(df) * 0.85)

train_data = scaled_data[:split_1]
val_data = scaled_data[split_1:split_2]
test_data = scaled_data[split_2:]

# Create sequences for each split
sequence_length = 30
X_train, y_train = create_sequences(train_data[:, :-1], train_data[:, -1], sequence_length)
X_val, y_val = create_sequences(val_data[:, :-1], val_data[:, -1], sequence_length)
X_test, y_test = create_sequences(test_data[:, :-1], test_data[:, -1], sequence_length)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(input_size=X_train.shape[2], hidden_size=50, num_layers=1, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for i in range(0, len(X_train), batch_size):
        inputs = X_train_tensor[i:i+batch_size].to(device)
        targets = y_train_tensor[i:i+batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).cpu().numpy()

# Rescale predictions
y_pred_rescaled = scaler.inverse_transform(
    np.hstack([y_pred, np.zeros((y_pred.shape[0], scaled_data.shape[1] - 1))])
)[:, 0]
y_test_rescaled = scaler.inverse_transform(
    np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))])
)[:, 0]

# Metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Calculate 0.5% accuracy
accuracy = [abs((pred - actual) / actual) <= 0.005 for pred, actual in zip(y_pred_rescaled, y_test_rescaled)]
accuracy_rate = sum(accuracy) / len(accuracy) * 100

# Directional Accuracy
actual_direction = np.diff(y_test_rescaled) > 0
predicted_direction = np.diff(y_pred_rescaled) > 0
directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

print(f"0.5% Accuracy: {accuracy_rate:.2f}%")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Visualization
plt.figure(figsize=(14, 8))

# Subplot 1: Actual vs Predicted Prices
plt.subplot(2, 1, 1)
plt.plot(y_test_rescaled, label="Actual", color="blue")
plt.plot(y_pred_rescaled, label="Predicted", color="red")
plt.title("LSTM: Actual vs Predicted Prices")
plt.legend()
plt.grid(True)

# Subplot 2: Prediction Accuracy
plt.subplot(2, 1, 2)
plt.scatter(range(len(accuracy)), accuracy, label=f"0.5% Accuracy: {accuracy_rate:.2f}%", color="green", s=5)
plt.title("Prediction Accuracy (1 = within 0.5%, 0 = outside 0.5%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()