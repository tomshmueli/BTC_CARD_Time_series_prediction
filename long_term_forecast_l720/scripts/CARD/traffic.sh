
# export CUDA_VISIBLE_DEVICES=0,1,2,3


if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=CARD


root_path_name=./dataset/traffic
data_path_name=traffic.csv
model_id_name=traffic_1
data_name=custom

random_seed=2021
for pred_len in 720 336 192 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
    --train_epochs 100\
      --patience 100 \
      --itr 1 --batch_size 4 --learning_rate 0.001 --merge_size 16 \
      --lradj CARD --warmup_epochs 20 
done
