import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import matplotlib.pyplot as plt

# 1. Feature Construction - Adding non-linear features
def create_features(df):
    """Create linear and non-linear features"""
    df_new = df.copy()
    
    # Original features
    numeric_features = df.select_dtypes(include=[np.number]).columns
    
    # Square terms
    for feature in numeric_features:
        df_new[f'{feature}_squared'] = df[feature] ** 2
        df_new[f'{feature}_cubed'] = df[feature] ** 3
    
    # Log terms (adding small constant to avoid log(0))
    for feature in numeric_features:
        if (df[feature] > 0).all():
            df_new[f'{feature}_log'] = np.log(df[feature] + 1)
    
    # Interaction terms between important features
    important_features = ['Adj Close', 'Volume', 'RSI']
    for i in range(len(important_features)):
        for j in range(i+1, len(important_features)):
            df_new[f'{important_features[i]}_{important_features[j]}_interaction'] = (
                df[important_features[i]] * df[important_features[j]]
            )
    
    return df_new

# Create lagged features
def create_lagged_features(data, lag=30):
    """
    Generate lagged features for all columns in the dataset for time series prediction.
    """
    lagged_frames = [data]  # Start with the original dataset
    for i in range(1, lag + 1):
        # Shift all columns in the dataset to create lagged features
        lagged_frame = data.shift(i).add_suffix(f'_Lag_{i}')
        lagged_frames.append(lagged_frame)
    
    # Combine all lagged dataframes
    df_lagged = pd.concat(lagged_frames, axis=1)
    df_lagged.dropna(inplace=True)  # Drop rows with NaN values due to lagging
    return df_lagged

# Load and preprocess the dataset
df = pd.read_csv('SP500_with_indicators_^GSPC.csv').dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Create enhanced feature set
df_enhanced = create_features(df)

# Define the features and target
features = ['Adj Close', 'Open', 'High', 'Low', 'Volume', 'ATR', 'ADX', 'RSI', 
            'MACD', 'MACD_Signal', 'Volatility', 'Max_Drawdown']
target = 'Adj Close'

# Apply lagged features
lagged_df = create_lagged_features(df_enhanced[features], lag=30)

# Prepare data for modeling
X = lagged_df.drop(columns=['Adj Close'])  # Exclude 'Adj Close' from features
y = lagged_df['Adj Close']  # Target is the 'Adj Close'

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training-validation and test sets
split_index = int(len(X) * 0.8)
X_train_val, X_test = X[:split_index], X[split_index:]
y_train_val, y_test = y[:split_index], y[split_index:]

# Hyperparameter tuning for Lasso
alphas = [0.0001, 0.001, 0.005, 0.01, 0.1, 0.5]  # List of alpha values to test

lasso = Lasso(max_iter=50000)

# Time Series Cross-Validation 
tscv = TimeSeriesSplit(n_splits=5)

# Track validation results for each parameter
validation_results = []
for alpha in alphas:
    fold_mae, fold_rmse, fold_r2 = [], [], []
    for train_idx, val_idx in tscv.split(X_train_val):
        X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
        
        # Train Lasso model
        lasso.alpha = alpha
        lasso.fit(X_train, y_train)
        
        # Predict on validation set
        y_val_pred = lasso.predict(X_val)
        
                # Calculate validation metrics
        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred)
        
        # Store metrics
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

   # Store and print average metrics for the current alpha
    avg_mae = np.mean(fold_mae)
    avg_rmse = np.mean(fold_rmse)
    avg_r2 = np.mean(fold_r2)
    
    print(f"Average metrics for alpha = {alpha}")
    print(f"MAE: {avg_mae:.4f}")
    print(f"RMSE: {avg_rmse:.4f}")
    print(f"R2: {avg_r2:.4f}")
    print("=" * 50)

# Select the best model based on validation MAE
best_result = min(validation_results, key=lambda x: x['MAE'])
best_alpha = best_result['alpha']

print("\nBest Hyperparameter (Time Series CV):")
print(f"Alpha: {best_alpha}, MAE: {best_result['MAE']:.2f}, RMSE: {best_result['RMSE']:.2f}, R2: {best_result['R2']:.2f}")

# Retrain the best model on the full training-validation set
lasso.alpha = best_alpha
lasso.fit(X_train_val, y_train_val)
lasso_pred = lasso.predict(X_test)

# Final evaluation metrics
mae = mean_absolute_error(y_test, lasso_pred)
rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
r2 = r2_score(y_test, lasso_pred)

# Accuracy within 0.5% error
lasso_accuracy = np.abs((lasso_pred - y_test) / y_test) <= 0.005
lasso_accuracy_rate = np.mean(lasso_accuracy) * 100

# Directional Accuracy
actual_direction = np.diff(y_test) > 0  # True if the actual value goes up
predicted_direction = np.diff(lasso_pred) > 0  # True if the predicted value goes up
directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

# Print results
print("\nFinal Evaluation:")
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2: {r2:.2f}')
print(f'Accuracy within 0.5%: {lasso_accuracy_rate:.2f}%')
print(f'Directional Accuracy: {directional_accuracy:.2f}%')

# Visualization
plt.figure(figsize=(12, 8))

# Lasso Regression: Actual vs Predicted
plt.subplot(2, 1, 1)
plt.plot(range(len(y_test)), y_test, label='Actual', color='blue')
plt.plot(range(len(lasso_pred)), lasso_pred, label='Lasso Predicted', color='red')
plt.title(f'Lasso Regression (L1 Regularization): Actual vs Predicted Prices')
plt.legend()
plt.grid(True)

# 0.5% Error Accuracy
lasso_accuracy_plot = [1 if abs((p - a) / a) <= 0.005 else 0 for p, a in zip(lasso_pred, y_test)]
lasso_accuracy_rate = sum(lasso_accuracy_plot) / len(lasso_accuracy_plot) * 100

# Plot the prediction accuracy
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 2)
plt.scatter(
    range(len(lasso_accuracy_plot)), 
    lasso_accuracy_plot, 
    label=f'Lasso Accuracy (0.5% error): {lasso_accuracy_rate:.2f}%', 
    color='green', s=5
)

# Add title, legend, and grid
plt.title('Prediction Accuracy (1 = within 0.5% error, 0 = outside 0.5% error)')
plt.xlabel('Test Sample Index')
plt.ylabel('Accuracy Indicator')
plt.legend()
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
