if [ ! -d "./logs/ShortForecasting/" ]; then
    mkdir -p ./logs/ShortForecasting/
fi

# Set W&B offline mode
export WANDB_MODE=offline

model_name=CARD

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Run the experiment for the Daily dataset only
python -u run.py \
--task_name short_term_forecast \
--is_training 1 \
--root_path ./dataset/m4 \
--seasonal_patterns 'Daily' \
--model_id m4_layer_2_Daily \
--model $model_name \
--data m4 \
--features M \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--dropout 0.0 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--batch_size 128 \
--d_model 128 \
--d_ff 512 \
--patch_len 16 \
--patience 400 \
--train_epochs 10 \
--stride 1 \
--n_heads 16 \
--top_k 5 \
--des 'Exp' \
--itr 1 \
--lradj CARD \
--learning_rate 0.0005 \
--loss 'SMAPE' --warmup_epochs 0 --merge_size 2 \
2>&1 | tee logs/ShortForecasting/$model_name'_'m4_layer_2_Daily.log &
