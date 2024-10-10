EXPERIMENT_DIR="experiments"

######## Configs ################

key="base_model_name | out_model_name | contrast_loss_type | beta | lora | best_response | optim | lr | num_epochs | lambda_sft | lambda_contrast | sft_type | reference_free | "

############ Table 3 ################

config1="Unbabel/TowerInstruct-7B-v0.2 | sft-xcomet_xl_xxl-inc7b-chosen-10lp-shuff-full | sigmoid | 0.1 | false | chosen | adamw_torch | 1e-5 | 1 | 1.0 | 0.0 | token | false | "
config1_args=" --create_pref --shuffle "

config2="Unbabel/TowerInstruct-7B-v0.2 | dpo-xcomet_xl_xxl-inc7b-10p-shuff-5e-7-from-base-sft | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | false | "   # the sft type is only set to map all configs but is not necessary
config2_args=" --eval_steps 800 --create_pref --shuffle "

config3="Unbabel/TowerInstruct-7B-v0.2 | dpo-xcomet_xl_xxl-inc7b-10lp-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 0.0 | 1.0 | token | false | "
config3_args=" --eval_steps 800 --create_pref --shuffle "

config4="/mnt/data/sagrawal/tower-alignment/experiments/sft-xcomet_xl_xxl-inc7b-chosen-10lp-shuff-full | dpo-xcomet_xl_xxl-inc7b-10p-shuff-1e-7-full-from-sft | sigmoid | 0.1 | false | chosen | rmsprop | 1e-7 | 3 | 0.0 | 1.0 | token | false | "   # the sft type is only set to map all configs but is not necessary
config4_args=" --eval_steps 800 --create_pref "

config5="Unbabel/TowerInstruct-7B-v0.2 | cpo-xcomet-xl_xxl-inc7b-10p-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "  
config5_args=" --create_pref --shuffle "

config6="Unbabel/TowerInstruct-13B-v0.2 | cpo-xcomet-xl_xxl-inc7b-13B-10p-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "  
config6_args=" --create_pref --shuffle "

config7="Unbabel/TowerInstruct-13B-v0.2 | sft-xcomet-xl_xxl-inc7b-13B-10p-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | adamw_torch | 1e-5 | 1 | 1.0 | 0.0 | token | false | " 
config7_args=" --create_pref --shuffle "

############ Table 5 ################

config8="Unbabel/TowerInstruct-7B-v0.2 | cpo-xcomet_xl_xxl-inc7b-6p-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "  
config8_args=" --train_lps en-ru,en-zh,en-de,ru-en,zh-en,de-en --create_pref --shuffle "

config9="Unbabel/TowerInstruct-7B-v0.2 | cpo-xcomet_kiwi-inc7b-6p-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "  
config9_args=" --train_lps en-ru,en-zh,en-de,ru-en,zh-en,de-en --create_pref --chosen_metric_name xcomet_kiwi --rejected_metric_name xcomet_kiwi --best_metric_name xcomet_kiwi --shuffle "

config10="Unbabel/TowerInstruct-7B-v0.2 | cpo-xcomet_kiwi-inc7b-10p-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "  
config10_args=" --create_pref --shuffle  --chosen_metric_name xcomet_kiwi --rejected_metric_name xcomet_kiwi --best_metric_name xcomet_kiwi "

config11="Unbabel/TowerInstruct-7B-v0.2 | cpo-alma-r-10p-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "   
config11_args="--dataset_name swetaagrawal/Alma-R-Pref-6lps"

config12="Unbabel/TowerInstruct-7B-v0.2 | cpo-alma-r-6p-1k-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "   
config12_args="--dataset_name swetaagrawal/Alma-R-Pref-6lps --max_per_lp 1000"

############ Figure 4 ################

config13="Unbabel/TowerInstruct-7B-v0.2 | cpo-xcomet-xl_xxl-inc7b-10p-200-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "   
config13_args="--create_pref --shuffle --max_per_lp 200 "

config14="Unbabel/TowerInstruct-7B-v0.2 | cpo-xcomet-xl_xxl-inc7b-10p-400-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "   
config14_args="--create_pref --shuffle --max_per_lp 400 "

config15="Unbabel/TowerInstruct-7B-v0.2 | cpo-xcomet-xl_xxl-inc7b-10p-600-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "   
config15_args="--create_pref --shuffle --max_per_lp 600 "

config16="Unbabel/TowerInstruct-7B-v0.2 | cpo-xcomet-xl_xxl-inc7b-10p-800-shuff-5e-7-full | sigmoid | 0.1 | false | chosen | rmsprop | 5e-7 | 3 | 1.0 | 1.0 | token | true | "   
config16_args="--create_pref --shuffle --max_per_lp 800 "


for run_id in {2..5} ; do

  # Set all vars 
  port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

  config=config${run_id}
  IFS=' | ' read -r -a var_names <<< "$key"
  IFS=' | ' read -r -a values <<< "${!config}"

  declare -A vars

  for ((i=0; $i<${#var_names[@]}; i++)); do
    vars[${var_names[i]}]=${values[i]}
  done;

  config_args=config${run_id}_args
  ARGS=${!config_args}

  if [ ${vars[reference_free]} == "true" ]; then ARGS="${ARGS} --reference_free "; fi

  # check other default values
  accelerate launch --main_process_port ${port} --config_file configs/deepspeed_config_zero3.yaml finetune.py --use_flash_attention_2 \
  --model_name_or_path ${vars[base_model_name]} --output_dir ${EXPERIMENT_DIR}/${vars[out_model_name]} --contrast_loss_type ${vars[contrast_loss_type]} \
  --gradient_accumulation_steps 16 --run_name ${vars[out_model_name]} --beta ${vars[beta]} --learning_rate ${vars[lr]} --best_response_key ${vars[best_response]} \
  --optimizer_type ${vars[optim]} --per_device_train_batch_size 1 --num_train_epochs ${vars[num_epochs]} ${ARGS} \
  --lambda_sft ${vars[lambda_sft]} --lambda_contrast ${vars[lambda_contrast]} --sft_type ${vars[sft_type]} 

done