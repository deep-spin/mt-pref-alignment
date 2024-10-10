
import os
import numpy as np
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from data_utils import load_comparison_dataset
from datasets import load_dataset
from data_utils import DPODataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from custom_trainer import *
import pandas as pd
from sklearn.metrics import confusion_matrix
current_device = Accelerator().local_process_index

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]], label_pad_token_id=-100, padding_value=0) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    concatenated_batch = {}

    max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            ).to(current_device)

    return concatenated_batch

def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id=-100,
    average_log_prob=False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        # normalized at the token level
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def get_batch_conditional_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id=-100,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_ps = logits.softmax(-1)

    entropy = -torch.sum(per_token_ps * torch.log(per_token_ps + 1e-10), dim=-1)
    return (entropy * mask).sum(-1)

def concatenated_forward(
        model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs =  {}
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        all_logps = get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
        )

        all_entropy = get_batch_conditional_entropy(
            all_logits,
            concatenated_batch["concatenated_labels"],
        )
        
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:len_chosen*2]

        entropy = (all_entropy[:len_chosen] + all_entropy[len_chosen:len_chosen*2]) / 2

        return chosen_logps, rejected_logps, entropy


def get_dataloader(tokenizer, prompt_choice="tower"):
    raw_dataset=load_dataset("sardinelab/MT-pref-human")["train"]
    dataset = load_comparison_dataset(tokenizer, "en-de,zh-en", raw_dataset,
                                        prompt_choice=prompt_choice, 
                                        best_response_key="chosen", max_prompt_length=256, max_length=512)

    data_collator = DPODataCollatorWithPadding(
                    tokenizer,
                    max_length=512,
                    max_prompt_length=256,
                    label_pad_token_id=-100,
                    padding_value=0,
                    truncation_mode="keep_end",
                    is_encoder_decoder=False,
                    max_target_length=128,
                )

    dataloader = DataLoader(dataset, shuffle=True, batch_size=4, collate_fn=data_collator )
    return dataloader

def get_model_predictions(model_name_or_path, dataloader):

    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True
        ).to(current_device)
    model.config.use_cache = False

    model_rewards_chosen = []
    model_rewards_rejected = []
    true_rewards_chosen = []
    true_rewards_rejected = []
    all_entropy = []
    prompt = []
    chosen = []
    rejected = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            chosen_logps, rejected_logps, entropy = concatenated_forward(model, batch)
        model_rewards_chosen.extend(chosen_logps.cpu().numpy())
        model_rewards_rejected.extend(rejected_logps.cpu().numpy())
        true_rewards_chosen.extend(batch['reward_chosen'])
        true_rewards_rejected.extend(batch['reward_rejected'])
        all_entropy.extend(entropy.cpu().numpy())
        prompt.extend(batch["prompt"])
        rejected.extend(batch["rejected_response_only"])
        chosen.extend(batch["chosen_response_only"])


    scores_df = pd.DataFrame({
        "model_chosen": model_rewards_chosen, 
        "model_rejected": model_rewards_rejected, 
        "da_chosen": true_rewards_chosen, 
        "da_rejected": true_rewards_rejected,
        "all_entropy": all_entropy})

    scores_df["model-diff"] = scores_df["model_chosen"] - scores_df["model_rejected"]
    scores_df["da-diff"] = scores_df["da_chosen"] - scores_df["da_rejected"] 

    return scores_df


def compute_expression(entropy, n):
    return max( entropy / np.log(n), 1e-9 )

def determine_phi_star(expressions):
    mean_expression = np.mean(np.log(expressions))
    phi_star = -1 / mean_expression
    return phi_star


model_name_or_path=sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
tokenizer.pad_token_id= 0

dataloader = get_dataloader(tokenizer, prompt_choice="alma")
scores_df = get_model_predictions(model_name_or_path, dataloader)
if model_name_or_path.startswith("Unbabel") or model_name_or_path.startswith("haoranxu"):
    model_name_or_path = "experiments/" + model_name_or_path
    os.makedirs(model_name_or_path, exist_ok=True)
scores_df.to_csv(model_name_or_path + "/model-acc.csv", index=None)

confusion = confusion_matrix(scores_df["da-diff"] >=0, scores_df["model-diff"] >= 0)
accuracy = confusion[1][1] / confusion[1].sum()

print(f"Model name: {model_name_or_path}    Accuracy: {accuracy}\n")
