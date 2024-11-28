import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv('SP500_with_indicators_^GSPC.csv').dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Compute percentage changes for all features
df_pct = df.pct_change().dropna()  # Calculate relative changes
target_column = 'Adj Close'

#Date,Open,High,Low,Close,Adj Close,Volume,ATR,ADX,RSI,MACD,MACD_Signal,Volatility,Max_Drawdown
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'ADX', 'RSI', 'MACD', 'MACD_Signal', 'Volatility', 'Max_Drawdown']
print(df_pct)

# Create lagged features for relative changes
def create_lagged_features(data, columns, lag=30):
    lagged_columns = {}
    for col in columns:
        for i in range(1, lag + 1):
            lagged_columns[f'{col}_Lag_{i}'] = data[col].shift(i)
    lagged_df = pd.concat([data, pd.DataFrame(lagged_columns, index=data.index)], axis=1)
    lagged_df.dropna(inplace=True)  # Drop rows with NaN values (due to lagging)
    return lagged_df

# Generate lagged features using percentage changes
lagged_df = create_lagged_features(df_pct, columns=features, lag=30)

# Separate features and target
X = lagged_df.drop(columns=['Adj Close']).values
y = lagged_df['Adj Close'].values  # Target is the percentage change in Adj Close

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Initialize XGBoost
xgb_model = xgb.XGBRegressor(
    max_depth=10,
    learning_rate=0.05,
    n_estimators=200,
    reg_alpha=0.01,
    reg_lambda=1.0,
    random_state=42,
)

# Cross-validation evaluation for all models
xgb_mae, xgb_rmse, xgb_r2 = [], [], []
fold = 1

for train_idx, test_idx in tscv.split(X_scaled):
    # Train-test split
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train the model
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # Append metrics
    xgb_mae.append(mean_absolute_error(y_test, xgb_pred))
    xgb_rmse.append(np.sqrt(mean_squared_error(y_test, xgb_pred)))
    xgb_r2.append(r2_score(y_test, xgb_pred))

    print(f"Fold {fold}:")
    print(f"  XGBoost - MAE={xgb_mae[-1]:.4f}, RMSE={xgb_rmse[-1]:.4f}, R2={xgb_r2[-1]:.4f}")
    fold += 1

# Average metrics across folds
print("\nCross-Validation Results (Average):")
print(f"XGBoost - MAE: {np.mean(xgb_mae):.4f}, RMSE: {np.mean(xgb_rmse):.4f}, R2: {np.mean(xgb_r2):.4f}")

# Final Evaluation on Test Set
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train the final model
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Convert actual and predictions to percentages
last_known_price = df.iloc[split_idx - 1]['Adj Close']  # Last known actual price
y_test_prices = [last_known_price]
for p in y_test:
    y_test_prices.append(y_test_prices[-1] * (1 + p))
y_pred_prices = [last_known_price]
for p in y_pred:
    y_pred_prices.append(y_pred_prices[-1] * (1 + p))

# Calculate metrics on absolute values
final_mae = mean_absolute_error(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_r2 = r2_score(y_test, y_pred)

print("\nFinal Evaluation (Absolute Values):")
print(f"XGBoost - MAE: {final_mae:.4f}, RMSE: {final_rmse:.4f}, R2: {final_r2:.4f}")

# Calculate Directional Accuracy
def directional_accuracy(y_actual, y_predicted):
    actual_directions = np.sign(y_actual[1:] - y_actual[:-1])  # Compute direction of actual changes
    predicted_directions = np.sign(y_predicted[1:] - y_predicted[:-1])  # Compute direction of predicted changes
    correct_directions = (actual_directions == predicted_directions).sum()  # Count matching directions
    accuracy = correct_directions / len(actual_directions) * 100  # Calculate percentage accuracy
    return accuracy

# Compute directional accuracy
directional_acc = directional_accuracy(y_test, y_pred)
print(f"\nDirectional Accuracy: {directional_acc:.2f}%")

# Calculate accuracy within 0.5%
def accuracy_within_threshold(y_actual, y_predicted, threshold=0.005):
    within_threshold = np.abs((y_actual - y_predicted) / y_actual) <= threshold
    accuracy = np.sum(within_threshold) / len(y_actual) * 100
    return accuracy

# Compute accuracy within 0.5%
accuracy_05 = accuracy_within_threshold(y_test, y_pred, threshold=0.005)
print(f"\nAccuracy within 0.5%: {accuracy_05:.2f}%")

# Plot actual vs predicted prices
plt.figure(figsize=(15, 8))
plt.plot(range(len(y_test_prices)), y_test_prices, label='Actual', color='blue', alpha=0.8)
plt.plot(range(len(y_pred_prices)), y_pred_prices, label='Predicted', color='red', alpha=0.8)
plt.title('S&P 500 Prices: Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Close')
plt.legend()
plt.grid(True)
plt.show()

# # 0.5% Error Accuracy Plot
plt.subplot(4, 1, 4)
xgb_accuracy_plot = [0.5 if abs((p - a) / a) <= 0.005 else 0 for p, a in zip(y_pred, y_test)]

plt.scatter(range(len(xgb_accuracy_plot)), xgb_accuracy_plot, label=f'XGBoost Accuracy (0.5% error): {accuracy_05:.2f}%', color='green', s=5)
plt.title('Prediction Accuracy (0.5 = within 0.5% error, 0 = outside 0.5% error)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()