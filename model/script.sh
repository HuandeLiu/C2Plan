
# --resume_checkpoint ckpts/openai_2025_08_20_16_56_22_947203/model036000.pt
#--resume_checkpoint  ckpts/openai_2025_08_20_16_58_21_924966/model024000.pt
MODEL_FLAGS="--dataset rplan --batch_size 100 --set_name train --target_set 5"
TRAIN_FLAGS="--lr 2e-4 --save_interval 10000 --weight_decay 0.05 --log_interval 1000 "

CUDA_VISIBLE_DEVICES='0' python image_train.py $MODEL_FLAGS $TRAIN_FLAGS --backup 1
#CUDA_VISIBLE_DEVICES='1' python image_sample.py $MODEL _FLAGS --model_path ckpts/exp/model250000.pt $SAMPLE_FLAGS
