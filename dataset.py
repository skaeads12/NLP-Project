
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from transformers import PreTrainedTokenizer

class HumanEvalDataset(Dataset):

    def __init__(
        self,
        data_dir: str = None,
        tokenizer: PreTrainedTokenizer = None,
    ):
        self.data = [json.loads(line) for line in open(data_dir, 'r').readlines()]
        self.PREFIX = "<fim_prefix>"
        self.SUFFIX = "<fim_suffix><fim_middle>"

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = self.PREFIX + sample["prompt"] + self.SUFFIX

        input_tokens = self.tokenizer(prompt, return_tensors="pt")

        return input_tokens
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = False):
        
        def collate_fn(batch):

            input_ids = pad_sequence([sample["input_ids"].squeeze(0) for sample in batch], batch_first=True, padding_value=self.tokenizer.eos_token_id, padding_side="left")
            attention_mask = pad_sequence([sample["attention_mask"].squeeze(0) for sample in batch], batch_first=True, padding_value=0, padding_side="left")

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# for test
if __name__=="__main__":
    
    from transformers import AutoTokenizer

    data_dir = "/workspace/nlp-project/data/humaneval.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")

    dataset = HumanEvalDataset(data_dir=data_dir, tokenizer=tokenizer)
    dataloader = dataset.get_dataloader(batch_size=32, shuffle=False)

    for batch in dataloader:
        print(batch["input_ids"].shape)
