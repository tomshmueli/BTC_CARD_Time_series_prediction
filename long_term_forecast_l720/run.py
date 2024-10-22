import argparse
import plot
import torch
import yaml
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.tools import load_config_from_file


def main_flow():
    setting = None
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

    # Set the experiment class for long-term forecasting
    Exp = Exp_Main

    if not args.is_predicting:
        # Output experiment settings
        print('Args in experiment:')
        print(args)

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
            print(f'>>>>>>>Testing only : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1)
            torch.cuda.empty_cache()

        return setting

    else:  # Predicting Task only
        # Output experiment settings
        print('Args in Prediction Task:')
        print(args)

        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
            args.distil, args.des, ii)

        exp = Exp(args)
        print(f'>>>>>>>Predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.predict(setting, load=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    run_setting = main_flow()
    print('Done')
