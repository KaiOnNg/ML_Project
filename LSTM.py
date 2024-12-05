from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
# Step 1: Add diagnostic imports and checks at top of file
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name()}")
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import copy

# Feature Construction (same as linear)
def create_features(df):
    """Create linear and non-linear features"""
    df_new = df.copy()
    numeric_features = df.select_dtypes(include=[np.number]).columns
    
    # Square and cube terms
    for feature in numeric_features:
        df_new[f'{feature}_squared'] = df[feature] ** 2
    
    # Log terms
    for feature in numeric_features:
        if (df[feature] > 0).all():
            df_new[f'{feature}_log'] = np.log(df[feature] + 1)
    
    # Interaction terms
    # important_features = ['Adj Close', 'Volume', 'RSI']
    # for i in range(len(important_features)):
    #     for j in range(i+1, len(important_features)):
    #         df_new[f'{important_features[i]}_{important_features[j]}_interaction'] = (
    #             df[important_features[i]] * df[important_features[j]]
    #         )
    return df_new

def create_sequences(data, target, sequence_length):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

def forward_stepwise_selection_optimized(X, y, sequence_length, model_fn, metric=mean_squared_error, improvement_threshold=0.01, max_features=None):
    """
    Optimized forward stepwise selection for features.

    Parameters:
        X (DataFrame): Input feature set.
        y (array): Target variable.
        sequence_length (int): Length of input sequences.
        model_fn (function): Function to create and train a model.
        metric (function): Metric to evaluate model performance (default: MSE).
        improvement_threshold (float): Minimum improvement to continue feature selection.
        max_features (int, optional): Maximum number of features to select.

    Returns:
        selected_features (list): List of selected features.
        performance_history (list): Performance history during selection.
    """
    remaining_features = list(X.columns)  # Features to select from
    selected_features = []  # Features already selected
    best_metric = float('inf')  # Initialize best metric
    performance_history = []  # Store feature selection progression

    # Create a single train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    while remaining_features:
        feature_metrics = {}

        for feature in remaining_features:
            # Candidate features include already selected + the new feature
            candidate_features = selected_features + [feature]

            # Create sequences for the candidate features
            X_candidate = X_train[candidate_features].to_numpy()
            X_seq, y_seq = create_sequences(X_candidate, y_train.values, sequence_length)

            X_val_candidate = X_val[candidate_features].to_numpy()
            X_val_seq, y_val_seq = create_sequences(X_val_candidate, y_val.values ,sequence_length)

            # Train model for the current subset of features
            model = model_fn(len(candidate_features))
            model.fit(X_seq, y_seq)

            # Predict and calculate the metric
            y_pred = model.predict(X_val_seq)
            feature_metrics[feature] = metric(y_val_seq, y_pred)

        # Select the best feature from the current round
        best_feature = min(feature_metrics, key=feature_metrics.get)
        best_feature_metric = feature_metrics[best_feature]

        # Check for improvement
        if best_feature_metric < best_metric - improvement_threshold:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_metric = best_feature_metric
            performance_history.append((len(selected_features), best_metric))
            print(f"Added feature: {best_feature} (Metric: {best_metric:.4f})")
        else:
            print("No significant improvement. Stopping selection.")
            break

        # Stop if maximum features reached
        if max_features and len(selected_features) >= max_features:
            break

    # Visualization
    if performance_history:
        features, metrics = zip(*performance_history)
        plt.figure(figsize=(10, 6))
        plt.plot(features, metrics, marker='o', linestyle='-', color='b')
        plt.title('Forward Stepwise Selection: Error vs. Number of Features')
        plt.xlabel('Number of Features')
        plt.ylabel('Error Metric')
        plt.grid(True)
        plt.show()

    print(f"Final Selected Features: {selected_features}")
    return selected_features, performance_history


# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # No dropout if num_layers=1
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Custom wrapper for LSTM compatible with GridSearchCV
class LSTMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, lr=0.001, epochs=50, batch_size=16, dropout=0.2, weight_decay=1e-4, patience=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = torch.device('cuda')
        self.model = None

    def _build_model(self):
        model = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout).to(self.device)
        return model

    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)

        # Create training and validation splits
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = self._build_model().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss = float('inf')
        patience_counter = 0
        best_model_state = copy.deepcopy(self.model.state_dict())


        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    val_loss += loss.item()

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Load best model
        self.model.load_state_dict(best_model_state)

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)

# Load and preprocess data
df = pd.read_csv('SP500_with_indicators_^GSPC.csv').dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.drop(columns=['Close'])

# Split data
split_idx = int(len(df) * 0.8)
df_train = df[:split_idx]
df_test = df[split_idx:]

# Feature construction
df_train = create_features(df_train)
df_test = create_features(df_test)

# Extract features and targets
X_train = df_train.drop(columns=['Adj Close'])
y_train = df_train['Adj Close']
X_test = df_test.drop(columns=['Adj Close'])
y_test = df_test['Adj Close']

# Run optimized forward selection
selected_features, performance_history = forward_stepwise_selection_optimized(
    X_train, 
    y_train, 
    sequence_length=30, 
    model_fn=LSTMRegressor, 
    metric=mean_squared_error, 
    improvement_threshold=0.01,  # Minimum improvement to continue
    max_features=10  # Optional: Limit the number of selected features
)

# Use only the selected features
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# Normalize data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create sequences
sequence_length = 30
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, sequence_length)

# Hyperparameter grid
param_grid = {
    'hidden_size': [50, 100, 200],
    'num_layers': [1, 2],
    'lr': [0.001, 0.01],
    'batch_size': [16, 32],
    'dropout': [0.2, 0.4],
    'weight_decay': [1e-4, 1e-3],
    'epochs': [50],  # Fixed epochs for GridSearch
}

# Custom scorer
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=LSTMRegressor(input_size=X_train_seq.shape[2]),
    param_grid=param_grid,
    scoring=scorer,
    cv=TimeSeriesSplit(n_splits=3),
    verbose=2,
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train_seq, y_train_seq) 

# Best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the final model on the entire training set
final_model = LSTMRegressor(input_size=X_train_seq.shape[2], **best_params)
final_model.fit(X_train_seq, y_train_seq)

# Test performance
lstm_pred = final_model.predict(X_test_seq)

# Calculate final metrics
mae = mean_absolute_error(y_test_seq, lstm_pred)
rmse = np.sqrt(mean_squared_error(y_test_seq, lstm_pred))
r2 = r2_score(y_test_seq, lstm_pred)

# Calculate accuracies
lstm_accuracy = np.abs((lstm_pred - y_test_seq) / y_test_seq) <= 0.005
lstm_accuracy_rate = np.mean(lstm_accuracy) * 100

actual_direction = np.diff(y_test_seq) > 0
predicted_direction = np.diff(lstm_pred.flatten()) > 0
directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

# Print final results
print("\nFinal Evaluation:")
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2: {r2:.2f}')
print(f'Accuracy within 0.5%: {lstm_accuracy_rate:.2f}%')
print(f'Directional Accuracy: {directional_accuracy:.2f}%')

# Visualization
plt.figure(figsize=(12, 8))

# LSTM: Actual vs Predicted
plt.subplot(2, 1, 1)
plt.plot(y_test, label='Actual', color='blue')
plt.plot(lstm_pred, label='LSTM Predicted', color='red')
plt.title('LSTM: Actual vs Predicted Prices')
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(2, 1, 2)
plt.scatter(range(len(lstm_accuracy)), 
           lstm_accuracy, 
           label=f'LSTM Accuracy (0.5% error): {lstm_accuracy_rate:.2f}%',
           color='green', s=5)
plt.title('Prediction Accuracy (1 = within 0.5% error, 0 = outside 0.5% error)')
plt.xlabel('Test Sample Index')
plt.ylabel('Accuracy Indicator')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()