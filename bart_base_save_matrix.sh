meta_path='PATH_TO_YOUR_PROJECT'
log_path='PATH_TO_TOUR_LOG'

# original
mkdir -p $log_path/


CUDA_VISIBLE_DEVICES=0 python $meta_path/run_summarization.py \
    --model_name_or_path $meta_path/models/bart-base-cnn \
    --do_predict \
    --save_similar_matrix \
    --similar_matrix_path $log_path/bart-base \
    --similar_matrix_per_size 1000 \
    --dataset_name cnn_dailymail \
    --dataset_config_name '3.0.0' \
    --dataset_cache_dir  'PATH_TO_CACHE_DIR' \
    --max_source_length 512 \
    --max_target_length 128 \
    --generation_num_beams 4 \
    --output_dir $log_path
