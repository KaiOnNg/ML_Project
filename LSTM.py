from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

# Custom Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, i):
        return (
            self.X[i:i+self.sequence_length],
            self.y[i+self.sequence_length-1]
        )

# Modified forward_stepwise_selection function
def forward_stepwise_selection(X, y, sequence_length, model_fn, max_features=10):
    selected_features = []
    remaining_features = list(X.columns)
    
    print("Forward stepwise selection:")
    for _ in range(max_features):
        best_score = float('inf')
        best_feature = None
        
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_subset = X[current_features].values
            
            # Create dataset and dataloader
            dataset = TimeSeriesDataset(X_subset, y.values, sequence_length)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Initialize model
            model = model_fn(len(current_features))
            optimizer = torch.optim.Adam(model.parameters())
            criterion = torch.nn.MSELoss()
            
            # Training loop
            model.train()
            for epoch in range(5):  # Reduced epochs for feature selection
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
            if epoch_loss < best_score:
                best_score = epoch_loss
                best_feature = feature
        
        if best_feature is None:
            break
            
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
    
    return selected_features
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

# Forward Stepwise Selection
# selected_features = forward_stepwise_selection(
#     df[features], target, sequence_length=30, model_fn=lambda input_size: LSTM(input_size).to("cpu"), max_features=10
# )
# print("Selected Features:", selected_features)

# selected_features = ['Adj Close', 'Open', 'High', 'Low', 'Volume', 'ATR', 'ADX', 'RSI', 
#             'MACD', 'MACD_Signal', 'Volatility', 'Max_Drawdown']

# selected_features = ['Volume', 'MACD', 'Adj Close_log']
# Use only selected features
# df = df[selected_features + ["Adj Close"]]

# Normalize data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df.drop(columns=["Adj Close"]).values)
y_scaled = scaler_y.fit_transform(df["Adj Close"].values.reshape(-1, 1))

# Split into train-validation-test sets
split_1 = int(len(df) * 0.7)
split_2 = int(len(df) * 0.85)

X_train_data = X_scaled[:split_1]
X_val_data = X_scaled[split_1:split_2]
X_test_data = X_scaled[split_2:]
y_train_data = y_scaled[:split_1]
y_val_data = y_scaled[split_1:split_2]
y_test_data = y_scaled[split_2:]

# Create sequences for each split
sequence_length = 30
X_train, y_train = create_sequences(X_train_data, y_train_data, sequence_length)
X_val, y_val = create_sequences(X_val_data, y_val_data, sequence_length)
X_test, y_test = create_sequences(X_test_data, y_test_data, sequence_length)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(input_size=X_train.shape[2], hidden_size=100, num_layers=1, dropout=0.4).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 40
batch_size = 16

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0

    for i in range(0, len(X_train), batch_size):
        inputs = X_train_tensor[i:i+batch_size].to(device)
        targets = y_train_tensor[i:i+batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = X_test_tensor[i:i+batch_size].to(device)
            targets = y_test_tensor[i:i+batch_size].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    # Calculate average losses
    avg_train_loss = train_loss / (len(X_train) / batch_size)
    avg_val_loss = val_loss / (len(X_test) / batch_size)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor.to(device)).cpu().numpy()

# Rescale predictions and targets
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Calculate 0.5% accuracy
accuracy = np.array([abs((pred - actual) / actual) <= 0.005 
                    for pred, actual in zip(y_pred.flatten(), y_test.flatten())])
accuracy_rate = np.mean(accuracy) * 100  # No need for explicit float conversion

# Directional Accuracy
actual_direction = np.diff(y_test.flatten()) > 0
predicted_direction = np.diff(y_pred.flatten()) > 0
directional_accuracy = np.mean(actual_direction == predicted_direction) * 100


print(f"0.5% Accuracy: {accuracy_rate:.2f}%")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Visualization
plt.figure(figsize=(15, 10))

# Subplot 1: Actual vs Predicted Prices
plt.subplot(2, 1, 1)
plt.plot(range(len(y_test)), y_test, label='Actual', color='blue')
plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red')
plt.title('LSTM: Actual vs Predicted Prices')
plt.legend()
plt.grid(True)

# Subplot 2: Prediction Accuracy (within 0.5% error)
plt.subplot(2, 1, 2)
plt.scatter(range(len(accuracy)), accuracy, label=f'LSTM Accuracy (0.5% error): {float(accuracy_rate):.2f}%', color='green', s=5)

plt.title('Prediction Accuracy (1 = within 0.5% error, 0 = outside 0.5% error)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()