# semantic-watermark
Code for paper: Watermarking Conditional Text Generation for AI Detection: Unveiling Challenges and a Semantic-Aware Watermark Remedy

![The outputs with the original watermark (OW) \citep{kirchenbauer2023watermark} and our proposed semantic-aware watermark (SW) on a test example from DART -- a data-to-text generation benchmark -- with parameters $\gamma=0.1$ and $\delta=5$. We expect $\sim$ 90\% of human-generated texts from the red list, whereas AI primarily utilizes the green list. Both watermarks yield high $z$-scores ($z>4$), indicating strong watermark strength for detection. Yet, OW forces the algorithm to generate from the red list due to randomly assigning key source entities (Mandy Patinkin) to it. As $\delta$ increases (towards a hard watermark), excluding these red tokens risks more hallucinations (words with underline).](./watermark_example.png)

# 1. Create Environment

Set the environment and install the required packages.

```
conda crate watermark
conda activate watermark
pip install -r requirements.txt
```
# 2. Download The Trained Backbone Models
We relased our traind models on those datasets. Part of the models were from huggingface and the rest of the models were trained by ourself.

Released Model can be found in [Google Drive]().


# 1.Create Semtantic Matrix
Use `bart_base_save_matrix.sh` to create the similar matrix first. (We use the L2 distance to compute the similarity between different tokens, you can also try to use other metrics, eg. cos-similarity). 
```python
meta_path='PATH_TO_YOUR_PROJECT'
log_path='PATH_TO_TOUR_LOG'

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
```
+ `meta_path` is the root dir of your current repo.  
+ `log_path` path to save your generated files.
+ `dataset_cache_dir` path to save your cache dir. (If you want to overwrite the cached dir, you can add `--overwrite_cache`. Sometimes the cached dataset may cause some mistakes.)

To conduct experiments on other models and datasets, please change the corresponding values, like `model_name_or_path`, `dataset_name`, etc. And if you want to use FLAN-T5 models, you have to add `--source_prefix "summarize: "` for CNN and Xsum. For WEB_NLG and DART datsets, we set `--source_prefix "translate Graph to English: "`.

# 2. Orignal Watermark

Use `bart_base_watermark.sh`, we can reproduce the original watermark method proposed by Kirchenbauer[1](https://arxiv.org/abs/2301.10226). Our method adapted from there method.
```python
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
```
We can set `gamma` and `delta` manually. For instance, `bash bart_base_watermark.sh 0.25 2` means we set `gamma=0.25` and `delta=2`.

# 3. Semantic Watermark
Use `bart_base_watermark_cluster.sh` to reproduce our semantic-watermark.

```python
meta_path='PATH_TO_YOUR_PROJECT'
log_path='PATH_TO_TOUR_LOG'

watermark_percent=$1
k=$2
watermark_delta=$3

sub_dir=watermark_${watermark_percent}_${watermark_delta}_cluster_total_${k}

# original
mkdir -p $log_path/$sub_dir


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
    --dataset_cache_dir 'PATH_TO_CACHE_DIR' \
    --use_watermark \
    --use_cluster_enhance \
    --watermark_cluster_k $k \
    --detect_watermark \
    --show_generated_result \
    --similar_matrix_path 'PATH_TO_CACHED_MATRIX' \
    --watermark_gamma $watermark_percent \
    --watermark_delta $watermark_delta \
    --output_dir $log_path/$sub_dir \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate 
```
+ `k` is the only hyper-parameter that introduced by our semantic-watermark, which controls the scope of semantic-related tokens used in this algorithm.
+ `similar_matrix_path` should be set as the same path created in `Create Semtantic Matrix`.
+ `deteck_watermark` was set to conduct the detection on the generated result.


We can set `gamma`, `k` and `delta` manually. For instance, `bash bart_base_watermark_cluster.sh 0.25 1 2` means we set `gamma=0.25`, `k=1` and `delta=2`.

# References
If you find this repository useful, please consider giving a star and citing this work:
```
@misc{fu2024watermarking,
      title={Watermarking Conditional Text Generation for AI Detection: Unveiling Challenges and a Semantic-Aware Watermark Remedy}, 
      author={Yu Fu and Deyi Xiong and Yue Dong},
      year={2024},
      eprint={2307.13808},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```