from vllm import LLM, SamplingParams
import pandas as pd
import argparse
import json

LANG_TABLE = {
    "de": "German", 
    "en": "English", 
    "zh": "Chinese", 
    "ru": "Russian",
    "cs": "Czech", 
    "ja": "Japanese",
    "pl": "Polish",
    "ru": "Russian",
    "ta": "Tamil",
    "nl": "Dutch",
    "fr": "French",
    "ko": "Korean",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
}

def read_file(fname):
    output = []
    with open(fname) as f:
        for line in f:
            output.append(line.strip())
    return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_or_name", type=str, default="haoranxu/ALMA-7B")
    parser.add_argument("-o", "--output_name", required=True, type=str)
    parser.add_argument("--max_new_tokens", default=1024, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--num_return_sequences", default=1, type=int)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--prompt_choice", default="tower", type=str) 
    parser.add_argument("--src_lang", required=True, type=str) 
    parser.add_argument("--tgt_lang", required=True, type=str) 
    parser.add_argument("--tsv-fname", required=True, type=str) 

    args = parser.parse_args()
    return args


def main(args):
    model_path_or_name = args.model_path_or_name
    output_name = args.output_name

    llm = LLM(model=f"{model_path_or_name}", trust_remote_code=True)
    sampling_params = SamplingParams(temperature=args.temperature, 
                                     max_tokens=args.max_new_tokens,
                                     top_p=args.top_p,
                                     top_k=args.top_k,
                                     use_beam_search=(args.num_beams > 1),
                                     n=args.num_return_sequences,
                                     best_of=args.num_beams)


    src_lang = args.src_lang
    tgt_lang = args.tgt_lang

    src_text = pd.read_csv(args.tsv_fname, sep="\t")["text"].to_list()

    if args.prompt_choice == "alma":
        prompts =  [f"Translate this from {LANG_TABLE[src_lang]} to {LANG_TABLE[tgt_lang]}:\n{LANG_TABLE[src_lang]}: {x}\n{LANG_TABLE[tgt_lang]}:" for x in src_text]
    else:
        prompts =  [f'<|im_start|>user\nTranslate the following {LANG_TABLE[src_lang]} source text to {LANG_TABLE[tgt_lang]}:\n{LANG_TABLE[src_lang]}: {x}\n{LANG_TABLE[tgt_lang]}: <|im_end|>\n<|im_start|>assistant\n' for x in src_text]

    outputs_raw= llm.generate(prompts, sampling_params)
    output = [{"out": x.outputs[0].text} for x in outputs_raw]

    with open(f"outputs/{output_name}.{src_lang}-{tgt_lang}.json", 'w') as fout:
        json.dump(output, fout)
        # for line in output:
            # fout.write(f"{line}\n")


if __name__ == "__main__":
    args = get_args()
    main(args)