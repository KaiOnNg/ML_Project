import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv('SP500_with_indicators_^GSPC.csv').dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Create lagged features
def create_lagged_features(data, lag=30):
    """
    Generate lagged features for time series prediction.
    """
    df_lagged = data.copy()
    for i in range(1, lag + 1):
        df_lagged[f'Lag_{i}'] = df_lagged['Adj Close'].shift(i)
    df_lagged.dropna(inplace=True)  # Drop rows with NaN values (due to lagging)
    return df_lagged

# Apply lagged features
lagged_df = create_lagged_features(df[['Adj Close']], lag=30)

# Prepare data for modeling
X = lagged_df.drop(columns=['Adj Close'])
y = lagged_df['Adj Close']

# Train-test split (80% train, 20% test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_scaled)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 1% Error Check
accuracy = [abs((pred - actual) / actual) <= 0.01 for pred, actual in zip(y_pred, y_test)]
accuracy_rate = sum(accuracy) / len(accuracy) * 100

# Visualization
plt.figure(figsize=(15, 10))

# Subplot 1: Actual vs Predicted Prices
plt.subplot(2, 1, 1)
plt.plot(range(len(y_test)), y_test, label='Actual', color='blue')
plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red')
plt.title('Linear Regression: Actual vs Predicted Prices')
plt.legend()
plt.grid(True)

# Subplot 2: Prediction Accuracy (within 1% error)
plt.subplot(2, 1, 2)
plt.plot(accuracy, label=f'Accuracy (within 1% error): {accuracy_rate:.2f}%', color='green')
plt.axhline(y=1, color='r', linestyle='--')
plt.title('Prediction Accuracy (1 = within 1% error, 0 = outside 1% error)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print metrics
print(f"Overall accuracy (within 1% error): {accuracy_rate:.2f}%")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Square Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")
