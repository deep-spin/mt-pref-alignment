output_name=${1:-"nllb-54b"}
model_path_or_name=${2:-"facebook/nllb-moe-54b"}
gpu_id=${3:-"6,7"}

CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang de --tgt_lang en --tsv-fname data_for_human_preference_collection/german.ppl_100-250.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang zh --tgt_lang en --tsv-fname data_for_human_preference_collection/chinese.ppl_300-500.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang ru --tgt_lang en --tsv-fname data_for_human_preference_collection/russian_ppl100-110.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang nl --tgt_lang en --tsv-fname data_for_human_preference_collection/dutch_ppl100-110.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang fr --tgt_lang en --tsv-fname data_for_human_preference_collection/french.ppl_100-120.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang it --tgt_lang en --tsv-fname data_for_human_preference_collection/italian.ppl-80-450.098-097.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang ko --tgt_lang en --tsv-fname data_for_human_preference_collection/korean_ppl50-100.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang pt --tgt_lang en --tsv-fname data_for_human_preference_collection/portuguese.ppl_100-110.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang es --tgt_lang en --tsv-fname data_for_human_preference_collection/spanish.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   

CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang en --tgt_lang de --tsv-fname data_for_human_preference_collection/english.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang en --tgt_lang zh --tsv-fname data_for_human_preference_collection/english.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang en --tgt_lang ru --tsv-fname data_for_human_preference_collection/english.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang en --tgt_lang nl --tsv-fname data_for_human_preference_collection/english.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang en --tgt_lang fr --tsv-fname data_for_human_preference_collection/english.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang en --tgt_lang it --tsv-fname data_for_human_preference_collection/english.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang en --tgt_lang ko --tsv-fname data_for_human_preference_collection/english.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang en --tgt_lang pt --tsv-fname data_for_human_preference_collection/english.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   
CUDA_VISIBLE_DEVICES=${gpu_id} python  generate_nllb.py --src_lang en --tgt_lang es --tsv-fname data_for_human_preference_collection/english.ppl_100-150.tsv --model_path_or_name ${model_path_or_name} --output_name ${output_name}   

