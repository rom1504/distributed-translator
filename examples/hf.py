import os
import pandas as pd
import torch
import argparse
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_parquet("/home/rom1504/metadata_0000.parquet")
sentences = df["caption"].tolist()
opus_mt_url="Helsinki-NLP/opus-mt-en-de"
#opus_mt_url="Helsinki-NLP/opus-mt-en-sv"


class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer_name):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence1 = sentences[index]

        tokens = self.tokenizer(sentence1, return_tensors="pt")

        return tokens


tokenizer = AutoTokenizer.from_pretrained(opus_mt_url)
model = AutoModelForSeq2SeqLM.from_pretrained(opus_mt_url)
model.to(device)
model.eval()


def custom_collate_fn(data):
    """
    Data collator with padding.
    """
    tokens = [sample["input_ids"][0] for sample in data]
    attention_masks = [sample["attention_mask"][0] for sample in data]

    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

    batch = {"input_ids": padded_tokens, "attention_mask": attention_masks}
    return batch

test_data = CaptionDataset(df, opus_mt_url)
test_dataloader = DataLoader(
    test_data,
    batch_size=50,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    collate_fn=custom_collate_fn,
)

with torch.no_grad():
    decoded_tokens = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        output_tokens = model.generate(**batch)
        #decoded_tokens += tokenizer.batch_decode(output_tokens.to("cpu"), skip_special_tokens=True)


