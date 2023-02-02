from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple
from ParEx import ParExTokenizer
from transformers import AutoTokenizer

class ParExDataLoader:
    def __init__(
        self, reconstruct_tokenizer_ck: str, extract_tokenizer_ck: str, extract_max_length: int, keyword_max_length: int):
        dataset = load_dataset('csv', data_files='data/data.csv')
        dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
        test_valid_dataset = dataset.pop('test')
        test_valid_dataset = test_valid_dataset.train_test_split(test_size=0.5, seed=42)
        dataset['valid'] = test_valid_dataset.pop('train')
        dataset['test'] = test_valid_dataset.pop('test')
        self.dataset = dataset
        self.parex_tokenizer = ParExTokenizer(reconstruct_tokenizer_ck)
        self.extract_tokenizer = AutoTokenizer.from_pretrained(extract_tokenizer_ck)
        self.extract_max_length = extract_max_length
        self.keyword_max_length = keyword_max_length

    def shift_tokens_right(self, input_ids: Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        tok = {}

        extract_tok = self.extract_tokenizer(encoded_inputs['Sent'], padding='max_length', truncation=True, max_length=self.extract_max_length, return_tensors='pt')
        tok['extract'] = extract_tok

        tok['keyword'] = self.parex_tokenizer(encoded_inputs['Keywords'], padding='max_length', truncation=True, max_length=self.keyword_max_length, return_tensors='pt')

        recon_tok_dec = self.parex_tokenizer(encoded_inputs['Sent'], padding='max_length', truncation=True, max_length=self.extract_max_length, return_tensors='pt')
        tok['decoder_labels'] = recon_tok_dec['input_ids']#.masked_fill_(recon_tok_dec['input_ids'] == 1, -100)
        pad_token_id = 1
        decoder_start_token_id = 2
        tok['decoder_input_ids'] = self.shift_tokens_right(recon_tok_dec['input_ids'], pad_token_id, decoder_start_token_id)
        return tok

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "valid", "test"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=24)
            )
        return res
