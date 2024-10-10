# 0. imports
import os
os.environ["WANDB_PROJECT"] = "tower-exps"

from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM

from trainer import POTrainer
from data_utils import prepare_comparison_dataset, create_preferences
from accelerate import Accelerator
from peft import PeftModel
from datasets import load_dataset

import torch
from transformers import set_seed
set_seed(42)

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    margin: Optional[float] = field(default=0.0, metadata={"help": "the margin for SimPO loss"})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    best_response_key: Optional[str] = field(
        default="best_response",
        metadata={"help": "which key for sft loss"},
    )
    train_lps: Optional[str] = field(
        default="pt-en,en-es,en-ru,en-zh,es-en,en-ko,en-de,en-nl,en-fr,ru-en,zh-en,nl-en,de-en,fr-en,en-it,ko-en,en-pt,it-en",
        metadata={"help": "language pairs used for evaluation separated by comma"},
    )
    dataset_name: Optional[str] = field(
        default="Unbabel/TowerAligned-v0.1",
        metadata={"help": "name of the dataset"},
    )
    raw_data_csv: Optional[str] = field(
        default="data/all_translations_with_scores.csv",
        metadata={"help": "name of the dataset"},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "name of the dataset"},
    )
    eval_lps: Optional[str] = field(
        default="en-de,zh-en",
        metadata={"help": "language pairs used for evaluation separated by comma"},
    )
    create_pref:Optional[bool] = field(
        default=False, 
        metadata={"help": "create preference data"})
    shuffle:Optional[bool] = field(
        default=False, 
        metadata={"help": "shuffle training data"})
    no_eval:Optional[bool] = field(
        default=False, 
        metadata={"help": "dont perform evaluation"})
    prompt_choice: Optional[str] = field(default="tower", metadata={"help": "choice of prompt"})
    chosen_metric_name: Optional[str] = field(default="xcomet_xl_xxl", metadata={"help": "choice of prompt"})
    rejected_metric_name: Optional[str] = field(default="xcomet_xl_xxl", metadata={"help": "choice of prompt"})
    best_metric_name: Optional[str] = field(default="xcomet_xl_xxl", metadata={"help": "choice of prompt"})
    remove_systems: Optional[str] = field(default="", metadata={"help": "systems to remove"})
    max_per_lp: Optional[int] = field(default=None, metadata={"help": "Max instances per LP"})
    
    # training parameters
    model_name_or_path: Optional[str] = field(
        default="haoranxu/ALMA-7B",
        metadata={"help": "the location of the SFT model name or path"},
    )
    peft_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "language pairs used for evaluation separated by comma"},
    )
    learning_rate: Optional[float] = field(default=5e-07, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=150, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "warmup ratio"})
    
    # https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L133
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=4, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    load_in_8bit:Optional[bool] = field(default=False, metadata={"help": "load model in 8 bit"})
    use_flash_attention_2:Optional[bool] = field(default=False, metadata={"help": "use flash attention"})
    low_cpu_mem_usage:Optional[bool] = field(default=False, metadata={"help": "use low cpu memory"})
    use_peft:Optional[bool] = field(default=False, metadata={"help": "use peft configuration"})
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
    lora_modules: Optional[str] = field(default="q_proj v_proj k_proj out_proj fc_in fc_out wte", metadata={"help": "the lora target modules"})

    # tower alignment specific flags
    contrast_loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type - sigmoid/hinge/ipo"})
    sft_type: Optional[str] = field(default="token", metadata={"help": "Whether to normalize sft at the batch or instance level"})
    generate_during_eval: Optional[bool] = field(default=False, metadata={"help": "Whether to generate during evaluation"})
    average_log_prob: Optional[bool] = field(default=False, metadata={"help": "Whether to generate during evaluation"})
    reference_free: Optional[bool] = field(default=False, metadata={"help": "Whether to use reference normalization"})
    lambda_contrast: Optional[float] = field(default=1.0, metadata={"help": "weight on dpo loss"})
    lambda_sft: Optional[float] = field(default=0.0, metadata={"help": "weight on supervised loss"})

    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "max number of training steps"}) # 4 epochs
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    resume_path: Optional[str] = field(default="", metadata={"help": "the directory path to resume training from"})

    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )

    run_name: Optional[str] = field(default="dpo", metadata={"help": "name of the run"})

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.use_flash_attention_2:
        model_kwargs = dict(
            attn_implementation="flash_attention_2",
        )
    else:
        model_kwargs = {}

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=script_args.load_in_8bit,
        cache_dir="experiments/",
        use_cache=False if script_args.gradient_checkpointing else True,
        **model_kwargs
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if not script_args.reference_free:
        model_ref = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=script_args.load_in_8bit,
            use_cache=False if script_args.gradient_checkpointing else True,
            **model_kwargs
        )
        model_ref.eval()
    else:
        model_ref = None

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    # 2. Load the paired dataset
    if script_args.create_pref:
        raw_dataset, _, _ = create_preferences(file_path=script_args.raw_data_csv,
                                         chosen_metric_name=script_args.chosen_metric_name,
                                         rejected_metric_name=script_args.rejected_metric_name,
                                         best_metric_name=script_args.best_metric_name,
                                         remove_systems=script_args.remove_systems.split(","))
    else:
        raw_dataset = load_dataset(script_args.dataset_name, cache_dir="./data_cache")["train"]
        
    # filter by length and select lps
    dataset = prepare_comparison_dataset(tokenizer=tokenizer,
        train_lps=script_args.train_lps, 
        raw_dataset=raw_dataset,
        prompt_choice=script_args.prompt_choice, 
        best_response_key=script_args.best_response_key,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        max_per_lp=script_args.max_per_lp)
    
    if script_args.no_eval:
        train_dataset = dataset.shuffle(seed=42)
        eval_dataset = None
    else:
        if script_args.eval_dataset_name is not None:
            eval_raw_dataset = load_dataset(script_args.evaldataset_name, cache_dir="./data_cache")["train"]
            eval_dataset = prepare_comparison_dataset(tokenizer=tokenizer,
                            train_lps=script_args.eval_lps, 
                            raw_dataset=eval_raw_dataset,
                            prompt_choice=script_args.prompt_choice, 
                            best_response_key=script_args.best_response_key,
                            max_prompt_length=script_args.max_prompt_length,
                            max_length=script_args.max_length,
                            max_per_lp=script_args.max_per_lp)
            train_dataset = dataset
        else:
            dataset = dataset.train_test_split(test_size=500)
            train_dataset, eval_dataset = dataset["train"], dataset["test"]
        
        if script_args.shuffle:
            train_dataset = train_dataset.shuffle(seed=42)
            eval_dataset = eval_dataset.shuffle(seed=42)
    
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="no" if script_args.no_eval else "steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        warmup_ratio=script_args.warmup_ratio,
        save_strategy="no",
        save_only_model=True,
        save_safetensors=True,
        adam_beta2=0.95,
    )
    if script_args.use_peft or script_args.load_in_8bit:
        # if a peft model is available and is to be finetuned
        if script_args.peft_model_id is not None:
            model = PeftModel.from_pretrained(model, script_args.peft_model_id)
            ## If still need to fine-tune
            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = True
        else:
            peft_config = LoraConfig(
                r=script_args.lora_r,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                target_modules=["down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
    else:
        peft_config = None

    # 5. initialize the trainer
    dpo_trainer = POTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        loss_type=script_args.contrast_loss_type, 
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        generate_during_eval=script_args.generate_during_eval,
        average_log_prob=script_args.average_log_prob,
        reference_free=script_args.reference_free,
        lambda_sft=script_args.lambda_sft,
        lambda_contrast=script_args.lambda_contrast,
        sft_type=script_args.sft_type,
        margin=script_args.margin,
    )

    # 6. train
    if script_args.resume_path == "":
        resume_path = False
    else:
        resume_path = script_args.resume_path
    dpo_trainer.train(resume_path)
    dpo_trainer.save_model(script_args.output_dir)
