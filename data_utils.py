from datasets import load_dataset
import datasets
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import concatenate_datasets
from datasets import Dataset
import pandas as pd
import numpy as np
import scipy.stats as ss

LANG_TABLE = {
    "de": "German", 
    "en": "English", 
    "zh": "Chinese", 
    "cs": "Czech", 
    "ja": "Japanese",
    "pl": "Polish",
    "ru": "Russian",
    "ta": "Tamil",
    "es": "Spanish", 
    "nl": "Dutch",
    "ko": "Korean",
    "pt": "Portuguese",
    "fr": "French",
    "it": "Italian",
}

FLORES_LANG_CODES = {
     "en": "eng_Latn",
     "de": "deu_Latn",
     "zh": "zho_Hans",
     "ru": "rus_Cyrl",
     "nl": "nld_Latn",
     "fr": "fra_Latn",
     "it": "ita_Latn",
     "ko": "kor_Hang",
     "pt": "por_Latn",
     "es": "spa_Latn",
}

TICO_MAP = {
    "en": "en",
    "es": "es-LA",
    "fr": "fr",
    "pt": "pt-BR", 
    "zh": "zh",
    "ru": "ru",
}

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_file(fname):
    output = []
    with open(fname) as f:
        for line in f:
            output.append(line.strip())
    return output

def write_to_file(data, fname):
    with open(fname, "w") as f:
        for line in data:
            f.write(line + "\n")

def write_to_pkl_file(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def clean_outputstring(output):
    out = output.split("\n")
    if out[0].strip() != "":
        return out[0].strip()
    elif out[1].strip() != "":
        return out[1].strip()
    else:
        return out[0]

def format_prompt(text, src_lang, tgt_lang, prompt_choice="alma"):
    if prompt_choice == "alma":
        query = f"Translate this from {LANG_TABLE[src_lang]} to {LANG_TABLE[tgt_lang]}:\n{LANG_TABLE[src_lang]}: {text}\n{LANG_TABLE[tgt_lang]}: " 
    elif prompt_choice == "tower":
        query = f'<|im_start|>user\nTranslate the following {LANG_TABLE[src_lang]} source text to {LANG_TABLE[tgt_lang]}:\n{LANG_TABLE[src_lang]}: {text}\n{LANG_TABLE[tgt_lang]}: <|im_end|>\n<|im_start|>assistant\n' 
    elif prompt_choice == "mistral":
        query = f'<s>[INST] Translate this text from {LANG_TABLE[src_lang]} to {LANG_TABLE[tgt_lang]}:\n{LANG_TABLE[src_lang]}[/INST]\n' 
    elif prompt_choice == "none":
        query = text 
    else:
        print("Incorrect prompt choice")
        exit()
    return query

def meet_length_requirements(prompt_tok, max_length):
    # if prompt is too long
    if len(prompt_tok) > max_length:
        return False
    # if prompt is too short
    elif len(prompt_tok) < 10 :
        return False
    return True 

def create_preferences(file_path="data/all_translations_with_scores.csv", 
                       chosen_metric_name="xcomet_xl_xxl", rejected_metric_name="xcomet_xl_xxl", 
                       best_metric_name="xcomet_xl_xxl", remove_systems=[""]):
    data = pd.read_csv(file_path)
    data["xcomet_kiwi"] = (data['Unbabel/XCOMET-XXL'] + data['Unbabel/wmt23-cometkiwi-da-xxl'])/2

    chosen_count = {x:0 for x in data.model.unique() if x not in remove_systems}
    rejected_count = {x:0 for x in data.model.unique() if x not in remove_systems}

    dataset = []
    for gr_name, gr_df in data[~data.model.isin(remove_systems)].groupby(["segment_id","lp", "source"]):
        _, lp, source = gr_name
        if lp == "ko-en" or lp == "zh-en":
            gr_df = gr_df[~gr_df.model.str.startswith("nllb")]

        gr_df = gr_df.sort_values(chosen_metric_name, ascending=False)
        mt_outs = gr_df["mt"].to_list()
        model_names = gr_df["model"].to_list()

        overall_best = gr_df.loc[gr_df[best_metric_name].idxmax()]
        chosen = gr_df.loc[gr_df[chosen_metric_name].idxmax()]
        rejected = gr_df.loc[gr_df[rejected_metric_name].idxmin()]
        chosen_count[chosen["model"]]+=1
        rejected_count[rejected["model"]]+=1
        all_outs = dict(gr_df[["model", "mt"]].values)
                
        if chosen_metric_name  == rejected_metric_name:
            all_outs_scores = dict(gr_df[["model", chosen_metric_name]].values)
            all_outs_scores = {k+"_score":v for k,v in all_outs_scores.items()}
        else:
            all_outs_chosen_scores = dict(gr_df[["model", chosen_metric_name]].values)
            all_outs_rejected_scores = dict(gr_df[["model", rejected_metric_name]].values)
            all_outs_scores = {k+"_score": (all_outs_chosen_scores[k] + all_outs_rejected_scores[k])/2 for k,v in all_outs_rejected_scores.items()}
        data_dict = dict(source=source, 
                            lp=lp,
                            chosen=chosen["mt"], 
                            rejected=rejected["mt"],
                            best_response=overall_best["mt"],
                            chosen_score=chosen[chosen_metric_name],
                            rejected_score=rejected[rejected_metric_name])
        data_dict.update(all_outs)
        data_dict.update(all_outs_scores)
        dataset.append(data_dict)
    
    dataset_hf = Dataset.from_list(dataset)
    return dataset_hf, chosen_count, rejected_count


def prepare_comparison_dataset(tokenizer, train_lps, raw_dataset, 
                            prompt_choice="tower", best_response_key="best_response", 
                            max_prompt_length=256, max_length=512, max_per_lp=None):
    train_lps = train_lps.split(",")
    raw_dataset = raw_dataset.filter(lambda x: x['lp'] in train_lps)

    if max_per_lp is not None:
        import pandas as pd
        df_pandas = pd.DataFrame(raw_dataset)

        new_df = []
        for _, gr_df in df_pandas.groupby("lp"):
            new_df.append(gr_df.sample(max_per_lp))
        df_final = pd.concat(new_df)
        raw_dataset = Dataset.from_pandas(df_final)
    
    dataset = datasets.Dataset.from_dict({
        "prompt": [
            format_prompt(text, lang_pair.split("-")[0], lang_pair.split("-")[1], prompt_choice)
                    for lang_pair, text in zip(raw_dataset["lp"], raw_dataset["source"])
                    ],
        "chosen": raw_dataset["chosen"],
        "rejected": raw_dataset["rejected"],
        "best_response": raw_dataset[best_response_key],
    })

    dataset = dataset.filter(lambda example: example["best_response"] is not None) # some outputs are None
    dataset = dataset.filter(lambda example: meet_length_requirements(tokenizer(example["prompt"], add_special_tokens=False).input_ids, max_prompt_length))
    dataset = dataset.filter(lambda example: meet_length_requirements(tokenizer(example["prompt"] + " " + example["chosen"], add_special_tokens=False).input_ids, max_length))
    dataset = dataset.filter(lambda example: meet_length_requirements(tokenizer(example["prompt"] + " " +  example["rejected"], add_special_tokens=False).input_ids, max_length))
    dataset = dataset.filter(lambda example: meet_length_requirements(tokenizer(example["prompt"] + " " +  example["best_response"], add_special_tokens=False).input_ids, max_length))
    
    return dataset

    
@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        best_response: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        best_response_tokens = self.tokenizer(best_response, add_special_tokens=False)

        eos_token_id = self.tokenizer.eos_token_id
        # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
        eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
        # attention mask these indices to eos_token_id
        new_attention_mask = [
            0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected
        eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_c = [
            0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        eos_indices_rejected = [i for i, x in enumerate(best_response_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p for i, p in enumerate(best_response_tokens["attention_mask"])
        ]
        best_response_tokens["attention_mask"] = new_attention_mask_r


        # add EOS token to end of prompt
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        best_response_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        best_response_tokens["attention_mask"].append(1)
        
        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]), len(best_response_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
            }
            best_response_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in best_response_tokens.items()}

        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        best_response_sequence_tokens = {k: prompt_tokens[k] + best_response_tokens[k] for k in best_response_tokens}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        best_response_sequence_tokens["labels"] = best_response_sequence_tokens["input_ids"][:]
        best_response_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
            "best_response": best_response_sequence_tokens,
        }.items():
            if toks is not None:
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens


        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected
        batch["best_response"] = prompt + best_response
        batch["best_response_only"] = best_response

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            best_response = feature["best_response"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected, best_response)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)

