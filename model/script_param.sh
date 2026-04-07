TEST_FLAGS="--dataset rplan --batch_size 512 --set_name eval --target_set 6"
SAMPLE_FLAGS="--batch_size 64 --num_samples 64"
# $1 ckpts/xx $2 model_id $3 back_up
base_path="../$1/$2" # ck/xx/modelid.pt
output_path="../outputs/$1/$2"
code_path="../"
if [ -n "$3" ]; then
  code_path="../outputs/$1/code"
  logger_path="../outputs/$1/log.txt"
fi
mkdir -p $output_path $code_path
#CUDA_VISIBLE_DEVICES='0' python ${code_path}/image_sample.py $TEST_FLAGS --model_path ${base_path} $SAMPLE_FLAGS --output_path $output_path --logger_path $logger_path  --model_id $2

