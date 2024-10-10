from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import argparse
import pandas as pd
from tqdm import tqdm
import torch

lang_dict = {
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_or_name", type=str, default='facebook/nllb-moe-54b')
    parser.add_argument("-o", "--output_name", required=True, type=str)
    parser.add_argument("--max_new_tokens", default=1024, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--src_lang", required=True, type=str) 
    parser.add_argument("--tgt_lang", required=True, type=str) 
    parser.add_argument("--tsv-fname", required=True, type=str) 

    args = parser.parse_args()
    return args

def main(args):
    model_name = args.model_path_or_name
    output_name = args.output_name

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                                  torch_dtype=torch.bfloat16,
                                                  device_map="auto",)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    src_lang = lang_dict[args.src_lang]
    tgt_lang = lang_dict[args.tgt_lang]
    
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)

    src_text = pd.read_csv(args.tsv_fname, sep="\t")["text"].to_list()

    output = []
    for text in tqdm(src_text):
        output.append(translator(text, max_length=args.max_new_tokens, num_beams=args.num_beams)[0]['translation_text'])

    with open(f"outputs/{src_lang}_{tgt_lang}_{output_name}.txt", 'w') as fout:
            for line in output:
                fout.write(f"{line}\n")

if __name__ == "__main__":
    args = get_args()
    main(args)
