#TEST_FLAGS="--dataset rplan --batch_size 512 --set_name eval "
#SAMPLE_FLAGS="--batch_size 512 --num_samples 512"
## 5
#MODEL_ID="model200000.pt"
#CUDA_VISIBLE_DEVICES='0' python image_sample.py $TEST_FLAGS --target_set 5 --model_path ckpts/openai_2025_09_08_19_26_08_072339/$MODEL_ID $SAMPLE_FLAGS --output_path outputhousediff5  --model_id $MODEL_ID
#


TEST_FLAGS="--dataset rplan --batch_size 512 --set_name eval "
SAMPLE_FLAGS="--batch_size 512 --num_samples 512"

#TODO Ours
MODEL_ID="model300000.pt"
CUDA_VISIBLE_DEVICES='1' python image_sample.py $TEST_FLAGS --target_set 8 --model_path ../ckpts/openai_2025_10_20_09_52_20_842787/$MODEL_ID $SAMPLE_FLAGS --output_path ../output/ours8  --model_id $MODEL_ID
#CUDA_VISIBLE_DEVICES='1' python model/image_sample.py $TEST_FLAGS --target_set 7 --model_path ../ckpts/openai_2025_10_17_16_42_12_265680/$MODEL_ID $SAMPLE_FLAGS --output_path output/ours7  --model_id $MODEL_ID
#CUDA_VISIBLE_DEVICES='1' python model/image_sample.py $TEST_FLAGS --target_set 6 --model_path ../ckpts/openai_2025_10_20_15_35_50_666411/$MODEL_ID $SAMPLE_FLAGS --output_path output/ours6  --model_id $MODEL_ID
#CUDA_VISIBLE_DEVICES='1' python model/image_sample.py $TEST_FLAGS --target_set 5 --model_path ../ckpts/openai_2025_10_25_10_47_08_113805/$MODEL_ID $SAMPLE_FLAGS --output_path output/ours5  --model_id $MODEL_ID
