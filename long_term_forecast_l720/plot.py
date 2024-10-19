import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from data_provider.data_loader import BTC_Dataset  # Import your dataset class
from utils.tools import load_config_from_file

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from data_provider.data_loader import BTC_Dataset  # Import your dataset class
from utils.tools import load_config_from_file


def plot_flow(settings=None, btc_dataset=None, predictions=None, scaler=None, trues=None):
    """
    Function to plot prediction results with real dates and scaling if needed.
    """
    # Step 1: Load configurations from config file
    config = load_config_from_file()

    # Step 2: Check if we are provided with predictions or if we need to load from npy files
    if settings is None:
        PRED_NAME = config.get('pred_file_name')
    else:
        cut = settings.find('CARD')
        PRED_NAME = settings[:cut]

    if predictions is None:
        folder_path = f"results/{settings}/"
        predictions = np.load(f"{folder_path}pred.npy")
        trues = np.load(f"{folder_path}true.npy")

    # If no btc_dataset is provided, load a new one
    if btc_dataset is None:
        btc_dataset = BTC_Dataset(
            root_path=config['root_path'],
            data_path=config['data_path'],
            target=config['target'],
            features=config.get('features'),
            scale=config.get('rescale', 1) == 1,
            timeenc=config['timeenc'],
            freq=config['freq']
        )
        btc_dataset.__read_data__()

    # If scaler is not passed, use the one from the btc_dataset
    if scaler is None:
        scaler = btc_dataset.scaler

    # Step 3: Calculate prediction range (mean, upper, lower bounds)
    test_size, pred_len, _ = predictions.shape
    seq_len = btc_dataset.seq_len
    pred_len = btc_dataset.pred_len

    # Initialize lists to store the prediction range
    mean_predictions = np.zeros(test_size)
    upper_bound = np.zeros(test_size)
    lower_bound = np.zeros(test_size)
    last_predicted_prices = np.zeros(test_size)  # New array to store last predicted price
    first_predicted_prices = np.zeros(test_size)  # New array to store first predicted price

    for i in range(test_size):
        daily_predictions = []
        for j in range(pred_len):
            if i + j < test_size:
                daily_predictions.append(predictions[i, j, 0])

        daily_predictions = np.array(daily_predictions)
        mean_predictions[i] = np.mean(daily_predictions)
        upper_bound[i] = np.max(daily_predictions)
        lower_bound[i] = np.min(daily_predictions)

        # Extract the last/first predicted value for each sequence
        first_predicted_prices[i] = predictions[i, 0, 0]
        last_predicted_prices[i] = predictions[i, -1, 0]

    # Step 4: Apply inverse scaling for predictions
    mean = scaler.mean_[0]
    std_dev = np.sqrt(scaler.var_[0])

    mean_predictions_rescaled = (mean_predictions * std_dev) + mean
    upper_bound_rescaled = (upper_bound * std_dev) + mean
    lower_bound_rescaled = (lower_bound * std_dev) + mean
    last_predicted_prices_rescaled = (last_predicted_prices * std_dev) + mean  # Inverse scaling for last prices
    first_predicted_prices_rescaled = (first_predicted_prices * std_dev) + mean  # Inverse scaling for first prices

    real_dates = btc_dataset.real_dates

    try:
        test_dates = real_dates[seq_len + pred_len - 1: - (
                    pred_len - 1)]  # model skips first seq_len-1 dates and last pred_len-1 dates

        # Step 5: Handle true values if provided (inverse scale them if needed)
        if trues is not None:
            # Reshape trues to (test_size,) and apply inverse scaling
            trues_single_value = trues[:, -1, 0]  # Extract the final true value in each sequence
            trues_rescaled = (trues_single_value * std_dev) + mean  # Inverse scaling

        # Step 6: Plot actual test prices and prediction bands
        plt.figure(figsize=(10, 6))
        if trues is not None:
            plt.plot(test_dates, trues_rescaled, label='Actual Prices', color='blue')

        plt.plot(test_dates, mean_predictions_rescaled, label='Mean Predicted Prices', color='red')
        plt.fill_between(test_dates, lower_bound_rescaled, upper_bound_rescaled, color='gray', alpha=0.3,
                         label='Prediction Range')

        # Step 7: Plot the green line for the last predicted price
        plt.plot(test_dates, last_predicted_prices_rescaled, label='Last Predicted Price', color='green',
                 linestyle='--')

        # Step 8: Plot the green line for the last predicted price
        plt.plot(test_dates, first_predicted_prices_rescaled, label='Last Predicted Price', color='purple',
                 linestyle='--')

    except Exception as e:
        print(f"Error while plotting: {e}")
        print("Ensure correct dataset configuration")

    plt.xlabel('Date')
    plt.ylabel('Bitcoin Price')
    plt.title(f'Actual vs Predicted Bitcoin Prices with Prediction Range (Test Data) - {PRED_NAME}')
    plt.xticks(test_dates[::config.get('time_print_ticks')], rotation=45)
    plt.legend()
    plt.tight_layout()

    # Step 7: Save the plot as a file
    if not os.path.exists("Price_Prediction"):
        os.mkdir("Price_Prediction")
    plt.savefig(f"Price_Prediction/{PRED_NAME}.png")

    # Optionally display the plot
    plt.show()


if __name__ == '__main__':
    plot_flow(settings=None)  # Pass any settings if needed
