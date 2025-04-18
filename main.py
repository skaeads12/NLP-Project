
import os
from tqdm import tqdm
from argparse import ArgumentParser

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from dataset import HumanEvalDataset

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-d", "--data_dir", type=str, default="/workspace/nlp-project/data/humaneval.jsonl")
    parser.add_argument("-s", "--save_dir", type=str, default="/workspace/nlp-project/outputs")
    parser.add_argument("-m", "--model", type=str, default="bigcode/starcoderbase-1b")

    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-ml", "--max_length", type=int, default=1024)

    return parser.parse_args()

def main(args):

    save_dir = args.save_dir
    if not save_dir.endswith("/"):
        save_dir += "/"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto")

    dataset = HumanEvalDataset(data_dir=args.data_dir, tokenizer=tokenizer)
    dataloader = dataset.get_dataloader(batch_size=args.batch_size, shuffle=False,)

    total = 0
    for batch in tqdm(dataloader):
        
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_length,
            )

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for response in responses:
            with open(save_dir + f"{total}.txt", 'w') as f:
                f.write(response)

            total += 1

if __name__=="__main__":

    args = parse_args()
    main(args)
