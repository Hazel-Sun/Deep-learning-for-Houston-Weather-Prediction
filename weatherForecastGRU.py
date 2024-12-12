import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import GRU, Dense, Input
from matplotlib.dates import MonthLocator, DateFormatter

# Hyperparameters
# Context and forecast length
TIME_STEP = 60  # Time steps to look into the past (context) [days]

# Global nn parameters
CHOSEN_MODEL = "GRU"    # Choose which model structure to use
EPOCHS = 50             # Training epochs
BATCH_SIZE = 16         # Batch size
DROPOUT = 0.2           # Dropout
LEARNING_RATE = 0.001   # Learning rate
HIDDEN_SIZE = 50        # Number of GRU units
NUM_LAYERS = 2          # Number of GRU layers

# Function to print progress
def print_progress(step):
    print(f"Progress: {step}...")

# Read data and print progress
print_progress("Reading data")
data_2019 = pd.read_csv('HoustonWeather/Houston,TX 2019-01-01 to 2019-12-31.csv')
data_2020 = pd.read_csv('HoustonWeather/Houston,TX 2020-01-01 to 2020-12-31.csv')
data_2021 = pd.read_csv('HoustonWeather/Houston,TX 2021-01-01 to 2021-12-31.csv')
data_2022 = pd.read_csv('HoustonWeather/Houston,TX 2022-01-01 to 2022-12-31.csv')
data_2023 = pd.read_csv('HoustonWeather/Houston,TX 2023-01-01 to 2023-12-31.csv')

# Merge training data and print progress
print_progress("Merging data")
train_data = pd.concat([data_2019, data_2020, data_2021, data_2022], ignore_index=True)

# Clean data and print progress
print_progress("Cleaning data")
train_data['datetime'] = pd.to_datetime(train_data['datetime'])
train_data.sort_values('datetime', inplace=True)
train_data.set_index('datetime', inplace=True)
train_data = train_data['temp'].values.reshape(-1, 1)

# Clean validation data
data_2023['datetime'] = pd.to_datetime(data_2023['datetime'])
val_data = data_2023['temp'].values.reshape(-1, 1)

# Normalize data and print progress
print_progress("Normalizing data")
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)

# Create training and validation datasets and print progress
print_progress("Creating datasets")
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data, TIME_STEP)

# Reshape data to fit GRU input and print progress
print_progress("Reshaping data")
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Build GRU model and print progress
print_progress("Building model")
model = Sequential()
model.add(Input(shape=(TIME_STEP, 1)))  # Modify model definition to resolve warning
if CHOSEN_MODEL == "GRU":
    model.add(GRU(HIDDEN_SIZE, return_sequences=True))
    for _ in range(NUM_LAYERS - 1):
        model.add(GRU(HIDDEN_SIZE, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

# Compile model and print progress
print_progress("Compiling model")
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model and print progress
print_progress("Training model")
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# Prepare prediction data and print progress
print_progress("Preparing prediction data")
combined_data = pd.concat([data_2022, data_2023], ignore_index=True)
combined_data['datetime'] = pd.to_datetime(combined_data['datetime'])
combined_data.sort_values('datetime', inplace=True)
combined_data.set_index('datetime', inplace=True)
prediction_data = combined_data['temp'].values.reshape(-1, 1)
prediction_data = scaler.transform(prediction_data)

X_pred, y_pred = create_dataset(prediction_data, TIME_STEP)

# Reshape prediction data to fit GRU input and print progress
print_progress("Reshaping prediction data")
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[1], 1)

# Make predictions and print progress
print_progress("Making predictions")
predictions = model.predict(X_pred)

# Inverse scaling of predictions and print progress
print_progress("Inverse scaling predictions")
predictions = scaler.inverse_transform(predictions)

# Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE)
mae = mean_absolute_error(data_2023['temp'][TIME_STEP + 1:], predictions[:len(data_2023) - TIME_STEP - 1])
mse = mean_squared_error(data_2023['temp'][TIME_STEP + 1:], predictions[:len(data_2023) - TIME_STEP - 1])
rmse = np.sqrt(mse)
r2 = r2_score(data_2023['temp'][TIME_STEP + 1:], predictions[:len(data_2023) - TIME_STEP - 1])

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² (R-squared): {r2}")

# Plot temperature predictions and save as figure1
print_progress("Plotting temperature predictions")
plt.figure(figsize=(12, 6))
plt.plot(combined_data.index[TIME_STEP + 1:], predictions, label='Predicted Temperature with GRU')
plt.plot(data_2023['datetime'], data_2023['temp'], label='Actual Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Prediction with GRU')
plt.legend()

# Set date format and interval
ax = plt.gca()
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.set_xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'))

# Ensure the results folder exists
if not os.path.exists('result'):
    os.makedirs('result')

# Save the temperature prediction plot
plt.savefig('result/temperature_prediction_GRU.png')
plt.show()

# Plot MAE, MSE, RMSE and save as figure2
print_progress("Plotting metrics")
plt.figure(figsize=(8, 6))
metrics = ['MAE', 'RMSE', 'R²']
values = [mae, rmse, r2]
bars = plt.bar(metrics, values, color=['blue', 'green', 'red'])
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', color='black') 

plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Prediction Metrics with GRU')

# Save the metrics plot
plt.savefig('result/prediction_metrics_GRU.png')
plt.show()

print_progress("Completed")

import json

# Output metrics in json
metrics_dict = {
    "Model": "GRU",
    "MAE": mae,
    "RMSE": rmse,
    "R²": r2
}

with open('result/error_metrics_GRU.json', 'w') as json_file:
    json.dump(metrics_dict, json_file)

print("json has been outputed")