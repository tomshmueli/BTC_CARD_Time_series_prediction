import argparse
import os
import torch
import yaml
from exp.exp_main import Exp_Main
import random
import numpy as np


def load_config_from_file(config_path="config.yaml"):
    """Load hyperparameters from a YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CARD Model for Long-Term Time Series Forecasting')

    if len(os.sys.argv) > 1:
        # (Argument parsing remains as is from the long-term script, keeping compatibility with CLI)
        parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
        parser.add_argument('--model', type=str, required=True, default='CARD',
                            help='model name, options: [CARD]')
        parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate')
        parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=96, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        # Add other necessary args from the long-term script
        args = parser.parse_args()

    else:
        # Load configuration from file if no args are provided
        config = load_config_from_file()
        args = argparse.Namespace(**config)

    # Seed setup
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # GPU setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Output experiment settings
    print('Args in experiment:')
    print(args)

    # Set the experiment class for long-term forecasting
    Exp = Exp_Main

    # Training loop
    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_ff, args.factor, args.embed,
                args.distil, args.des, ii)

            exp = Exp(args)
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            torch.cuda.empty_cache()

    # Testing phase
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
            args.distil, args.des, ii)

        exp = Exp(args)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
