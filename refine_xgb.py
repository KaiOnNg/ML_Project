import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.utils import resample

def create_features(df):
    """Create linear and non-linear features"""
    
    # Original features
    numeric_features = df.select_dtypes(include=[np.number]).columns
    
    new_features = {}
    for feature in numeric_features:
        new_features[f'{feature}_squared'] = df[feature] ** 2
        new_features[f'{feature}_cubed'] = df[feature] ** 3
    
    # Log terms (adding small constant to avoid log(0))
    for feature in numeric_features:
        if (df[feature] > 0).all():
            new_features[f'{feature}_log'] = np.log(df[feature] + 1)
    
    # Interaction terms between important features
    important_features = ['Adj Close', 'Volume', 'RSI']
    for i in range(len(important_features)):
        for j in range(i+1, len(important_features)):
            new_features[f'{important_features[i]}_{important_features[j]}_interaction'] = (
                df[important_features[i]] * df[important_features[j]]
            )
    
    new_features_df = pd.DataFrame(new_features)
    df_new = pd.concat([df, new_features_df], axis=1)
    
    return df_new

def create_sequences(data, sequence_length):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])  # First column is Adj Close
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
    
    while remaining_features:
        feature_metrics = {}

        for feature in remaining_features:
            # Candidate features include already selected + the new feature
            candidate_features = selected_features + [feature]

            # Create sequences for the candidate features
            X_candidate = X_train[candidate_features].to_numpy()
            X_seq, y_seq = create_sequences(X_candidate, sequence_length)

            X_val_candidate = X_val[candidate_features].to_numpy()
            X_val_seq, y_val_seq = create_sequences(X_val_candidate, sequence_length)

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

# Load and preprocess the dataset
df = pd.read_csv('SP500_with_indicators_^GSPC.csv').dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Compute percentage changes for all features
df_pct = df.pct_change().dropna()  # Calculate relative changes
target_column = 'Adj Close'

#Date,Open,High,Low,Close,Adj Close,Volume,ATR,ADX,RSI,MACD,MACD_Signal,Volatility,Max_Drawdown
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'ADX', 'RSI', 'MACD', 'MACD_Signal', 'Volatility', 'Max_Drawdown']

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
construction_features = create_features(lagged_df)

# Separate features and target
X = construction_features.drop(columns=['Adj Close']).values
y = construction_features['Adj Close'].values  # Target is the percentage change in Adj Close

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Final Evaluation on Test Set
split_idx = int(len(X_scaled) * 0.8)
X_train_val, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train_val, y_test = y[:split_idx], y[split_idx:]

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Define the parameter grid
param_grid = {
    'max_depth': [5, 7, 10],                # Depth of the trees
    'learning_rate': [0.01, 0.05, 0.1],   # Step size for weight updates
    'n_estimators': [200, 300],      # Number of boosting rounds
    'reg_lambda': [1, 1.5, 2],             # L2 regularization
}

# Initialize XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=2,
    n_jobs=-1  # Use all available cores
)

grid_search.fit(X_train_val, y_train_val)
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
print(f"Best RMSE: {np.sqrt(-grid_search.best_score_)}")
print(f"Best MAE: {mean_absolute_error(y_test, grid_search.best_estimator_.predict(X_test))}")
print(f"Best R2: {r2_score(y_test, grid_search.best_estimator_.predict(X_test))}")
optimized_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)

# Train the final model
optimized_xgb_model.fit(X_train_val, y_train_val)
y_pred = optimized_xgb_model.predict(X_test)

# Convert actual and predictions to percentages
last_known_price = df.iloc[split_idx - 1]['Adj Close']  # Last known actual price
y_test_prices = [last_known_price]
for p in y_test:
    y_test_prices.append(y_test_prices[-1] * (1 + p))
y_pred_prices = [last_known_price]
for p in y_pred:
    y_pred_prices.append(y_pred_prices[-1] * (1 + p))

# Directional Accuracy
actual_direction = np.diff(y_test) > 0  # True if the actual value goes up
predicted_direction = np.diff(y_pred) > 0  # True if the predicted value goes up
directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
print(f"\nDirectional Accuracy: {directional_accuracy:.4f}%")

# Calculate accuracy within 0.5%
def accuracy_within_threshold(y_actual, y_predicted, threshold=0.005):
    within_threshold = np.abs((y_actual - y_predicted) / y_actual) <= threshold
    accuracy = np.sum(within_threshold) / len(y_actual) * 100
    return accuracy

# Compute accuracy within 0.5%
accuracy_05 = accuracy_within_threshold(y_test, y_pred, threshold=0.005)
print(f"\nAccuracy within 0.5%: {accuracy_05:.2f}%")

# calculate confidence interval of 95%
confidence = 0.95
errors = y_test - y_pred
mean_error = np.mean(errors)
std_error = np.std(errors)
margin_of_error = std_error * 1.96
confidence_interval = (mean_error - margin_of_error, mean_error + margin_of_error)
print(f"\n95% Confidence Interval: {confidence_interval}")

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