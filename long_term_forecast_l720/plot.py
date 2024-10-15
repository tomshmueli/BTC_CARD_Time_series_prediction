import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_provider.data_loader import BTC_Dataset  # Import your dataset class
from run import load_config_from_file

# Step 1: Load configuration from YAML
config = load_config_from_file()
PRED_NAME = 'pred'  # use the name of the pred.npy file you want to plot (inside results/npy_results/)

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

# Step 3: Read data and fit scaler (scaling only the 'Price' column now)
btc_dataset.__read_data__()

# Step 4: Load predictions from the .npy file (predictions on the test set)
predictions = np.load(f"results/npy_results/{PRED_NAME}.npy")
actual_size = predictions.shape[0]  # Get the actual size of the predictions
predictions_flat = predictions.reshape(-1, 1)  # Flatten predictions

# Step 5: Inverse transform predictions back to original scale
if config.get('features') == 'MS':
    # Create a dummy array with two columns for inverse transform (Price and Volume)
    dummy_input = np.zeros((predictions_flat.shape[0], 2))
    dummy_input[:, 0] = predictions_flat.flatten()  # Fill the first column with predicted 'Price'

    # Inverse transform using the StandardScaler (applies only to Price)
    predictions_rescaled = btc_dataset.inverse_transform(dummy_input)[:, 0]  # Extract only the Price column
else:
    # If 'S' (single feature), directly inverse transform
    predictions_rescaled = btc_dataset.inverse_transform(predictions_flat).flatten()

# Step 6: Load actual prices and dates from CSV
actual_prices = pd.read_csv(f"{config['root_path']}/{config['data_path']}")['Price']
dates = pd.read_csv(f"{config['root_path']}/{config['data_path']}")['date']  # Assuming a 'date' column

# Step 7: Adjust for the test portion (last 10% of data)
num_test = actual_size  # Adjust for your test split (20%)
test_prices = actual_prices[-num_test:]  # Extract test prices
test_dates = dates[-num_test:]  # Extract test dates

# Step 8: Plot actual test prices and predictions
plt.figure(figsize=(10, 6))
plt.plot(test_dates, test_prices, label='Actual Prices', color='blue')
plt.plot(test_dates, predictions_rescaled[:num_test], label='Predicted Prices', color='red')

# Step 9: Configure plot with dates
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title(f'Actual vs Predicted Bitcoin Prices (Test Data) - {PRED_NAME}')
plt.xticks(test_dates[::300], rotation=45)  # Adjust the step size for dates based on your data
plt.legend()
plt.tight_layout()

# Save the plot as a file (e.g., PNG)
if not os.path.exists("Price_Prediction"):
    os.mkdir("Price_Prediction")
plt.savefig(f"Price_Prediction/bitcoin_prediction_plot_{PRED_NAME}.png")

# Optionally, you can also display it if needed
plt.show()
