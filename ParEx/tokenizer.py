from transformers import AutoTokenizer

class ParExTokenizer:
    def __init__(self, tokenizer_ckpt: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
        self.tokenizer.add_tokens(["<k>", "</k>"])

    def __call__(self, sentences, **kwargs):
        return self.tokenizer(sentences, **kwargs)

    def save_pretrained(self, path):
        self.tokenizer.save_pretrained(path)

    def batch_decode(self, ids, **kwargs):
        return self.tokenizer.batch_decode(ids, **kwargs)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id