import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from data_provider.data_loader import BTC_Dataset  # Import your dataset class
from run import load_config_from_file

# Step 1: Load configuration from YAML
config = load_config_from_file()
PRED_NAME = 'hourly_500'  # Use the name of the pred.npy file you want to plot (inside results/npy_results/)

# Step 2: Initialize BTC_Dataset and load data using parameters from config
btc_dataset = BTC_Dataset(
    root_path=config['root_path'],
    data_path=config['data_path'],
    target=config['target'],
    features=config.get('features'),  # S --> Single feature mode / 'MS' --> Multi-feature mode
    scale=config.get('rescale', 1) == 1,  # Use rescale from config, treating 1 as True for scaling
    timeenc=1,  # Assuming time encoding is needed
    freq=config['freq']  # Assuming this is the test set
)

# Step 3: Read data and retrieve scaler parameters
btc_dataset.__read_data__()
scaler = btc_dataset.scaler  # Assuming scaler was fitted during data loading

# Step 4: Load predictions from the .npy file (predictions on the test set)
predictions = np.load(f"results/npy_results/{PRED_NAME}.npy")

# Step 5: Calculate prediction range (mean, upper, lower bounds)
test_size, pred_len, _ = predictions.shape  # Get shape: (testset size, pred_len, #features)

# Initialize lists to store the prediction range
mean_predictions = np.zeros(test_size)
upper_bound = np.zeros(test_size)
lower_bound = np.zeros(test_size)

# Calculate mean, upper, and lower bounds
for i in range(test_size):
    daily_predictions = []
    for j in range(pred_len):
        if i + j < test_size:
            daily_predictions.append(predictions[i, j, 0])

    daily_predictions = np.array(daily_predictions)
    mean_predictions[i] = np.mean(daily_predictions)
    upper_bound[i] = np.max(daily_predictions)  # or use np.percentile(daily_predictions, 90)
    lower_bound[i] = np.min(daily_predictions)  # or use np.percentile(daily_predictions, 10)

# Step 6: Manually apply inverse scaling (assuming scaler has mean_ and var_)
mean = scaler.mean_[0]  # Get the mean from the scaler (for Price)
std_dev = np.sqrt(scaler.var_[0])  # Get the standard deviation (for Price)

# Inverse scaling for mean, upper, and lower bounds
mean_predictions_rescaled = (mean_predictions * std_dev) + mean
upper_bound_rescaled = (upper_bound * std_dev) + mean
lower_bound_rescaled = (lower_bound * std_dev) + mean

try:
    # Step 7: Load actual prices and dates from CSV
    actual_prices = pd.read_csv(f"{config['root_path']}/{config['data_path']}")['Price']
    dates = pd.read_csv(f"{config['root_path']}/{config['data_path']}")['date']  # Assuming a 'date' column

    # Step 8: Adjust for the test portion (last 10% of data)
    num_test = test_size  # Adjust for your test split (20%)
    test_prices = actual_prices[-num_test:]  # Extract test prices
    test_dates = dates[-num_test:]  # Extract test dates

    # Step 9: Plot actual test prices and prediction bands (mean, upper, lower)
    plt.figure(figsize=(10, 6))
    plt.plot(test_dates, test_prices, label='Actual Prices', color='blue')
    plt.plot(test_dates, mean_predictions_rescaled, label='Mean Predicted Prices', color='red')
    plt.fill_between(test_dates, lower_bound_rescaled, upper_bound_rescaled, color='gray', alpha=0.3, label='Prediction Range')

except Exception as e:
    print(f"Error: {e}")
    print("Make sure to load from config correct dataset! hourly/daily")

# Step 10: Configure plot with dates
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title(f'Actual vs Predicted Bitcoin Prices with Prediction Range (Test Data) - {PRED_NAME}')
plt.xticks(test_dates[::300], rotation=45)  # Adjust the step size for dates based on your data
plt.legend()
plt.tight_layout()

# Save the plot as a file (e.g., PNG)
if not os.path.exists("Price_Prediction"):
    os.mkdir("Price_Prediction")
plt.savefig(f"Price_Prediction/{PRED_NAME}_bitcoin_prediction_plot_range.png")

# Optionally, you can also display it if needed
plt.show()
