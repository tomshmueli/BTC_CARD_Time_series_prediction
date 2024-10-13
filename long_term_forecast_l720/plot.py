import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from data_provider.data_loader import BTC_Dataset  # Import your dataset class
from run import load_config_from_file

# Step 1: Load configuration from YAML
config = load_config_from_file()
PRED_NAME = 'pred_s_hourly'  # Name of the prediction file to load

# Step 2: Initialize BTC_Dataset and load data using parameters from config
# Since we're now working with only 'Price', the 'features' is 'S'
btc_dataset = BTC_Dataset(
    root_path=config['root_path'],
    data_path=config['data_path'],
    target=config['target'],
    features='S',  # Single feature mode
    scale=config.get('rescale', 1) == 1,  # Use rescale from config, treating 1 as True for scaling
    timeenc=1,  # Assuming time encoding is needed
    freq=config['freq']  # Assuming this is the test set
)

# Step 3: Read data and fit scaler (scaling only the 'Price' column now)
btc_dataset.__read_data__()

# Step 4: Load the predicted data from the .npy file (use proper config key for results path)
predictions = np.load(f"results/npy_results/{PRED_NAME}.npy")

# Step 5: Flatten the predictions (since it's (640, 30, 1), we flatten it to a 1D array)
predictions_flat = predictions.reshape(-1, 1)

# Step 6: Inverse transform the predictions directly (only 'Price' is scaled now)
# Now we only need to inverse transform the 'Price' column (single column scaling)
predictions_rescaled = btc_dataset.inverse_transform(predictions_flat)

# Step 7: Flatten the rescaled predictions again for plotting
predictions_rescaled_flat = predictions_rescaled.flatten()

# Step 8: Load actual prices from CSV and slice to match prediction length
actual_prices = pd.read_csv(f"{config['root_path']}/{config['data_path']}")['Price']
actual_prices_trimmed = actual_prices[len(actual_prices) - len(predictions_rescaled_flat):]

# Step 8.5: Slice the predicted prices to match the length of actual prices
predictions_rescaled_flat = predictions_rescaled_flat[:len(actual_prices_trimmed)]  # Slice the predictions

# Step 9: Plotting actual vs predicted prices
plt.plot(actual_prices_trimmed, label='Actual Prices', color='blue')
plt.plot(predictions_rescaled_flat, label='Predicted Prices', color='red')

# Add labels and legend
plt.xlabel('Time Steps')
plt.ylabel('Bitcoin Price')
plt.title('Actual vs Predicted Bitcoin Prices')
plt.legend()

# Save the plot as a file (e.g., PNG)
if not os.path.exists("Price_Prediction"):
    os.mkdir("Price_Prediction")
plt.savefig(f"Price_Prediction/bitcoin_prediction_plot_single_feature_{PRED_NAME}.png")  # Saves the plot in the current directory as a PNG file

# Optionally, you can also display it if needed
plt.show()
