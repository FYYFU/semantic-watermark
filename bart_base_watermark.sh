watermark_percent=$1
watermark_delta=$2

meta_path='PATH_TO_YOUR_PROJECT'
log_path='PATH_TO_TOUR_LOG'

mkdir -p $log_path

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --use_env \
    --master_port 22333 \
    $meta_path/run_summarization.py \
    --model_name_or_path $meta_path/models/bart-base-cnn \
    --do_predict \
    --dataset_name cnn_dailymail \
    --dataset_config_name '3.0.0' \
    --max_source_length 512 \
    --max_target_length 128 \
    --generation_num_beams 4 \
    --use_watermark \
    --detect_watermark \
    --show_generated_result \
    --watermark_gamma $watermark_percent \
    --watermark_delta $watermark_delta \
    --dataset_cache_dir 'PATH_TO_CACHE_DIR' \
    --overwrite_cache \
    --output_dir  $log_path \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate | tee $log_path/log.txt
