import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from ParEx import ParExModel, ParExTokenizer, ParExConfig
import evaluate
import numpy as np

class LitParEx(pl.LightningModule):
    def __init__(
        self, extract_model_ck: str, reconstruct_model_ck: str, tokenizer_ck: str, lr: float,
        num_beams: int, max_length: int
    ):
        super(LitParEx, self).__init__()
        self.parex = ParExModel(extract_model_ck, reconstruct_model_ck)
        self.loss = nn.CrossEntropyLoss(ignore_index=1)
        self.valid_metric = evaluate.load("sacrebleu")
        self.test_metric = evaluate.load("sacrebleu")
        self.tokenizer = ParExTokenizer(tokenizer_ck)
        self.lr = lr
        self.num_beams = num_beams
        self.max_length = max_length
        self.save_hyperparameters()

    def export_model(self, path):
        self.parex.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def __postprocess(self, predictions, labels):
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        return decoded_preds, decoded_labels

    def training_step(self, batch, batch_idx):
        labels = batch.pop('decoder_labels')
        logits = self.parex(batch).logits

        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1).long()
        loss = self.loss(logits, labels)

        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('decoder_labels')
        preds = self.parex.generate(batch, num_beams=self.num_beams, max_length=self.max_length)

        decoded_preds, decoded_labels = self.__postprocess(preds, labels)
        self.valid_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def validation_epoch_end(self, outputs):
        results = self.valid_metric.compute()
        self.log('valid/bleu', results['score'], on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('decoder_labels')
        preds = self.parex.generate(batch, num_beams=self.num_beams, max_length=self.max_length)

        decoded_preds, decoded_labels = self.__postprocess(preds, labels)
        self.test_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def test_epoch_end(self, outputs):
        results = self.test_metric.compute()
        self.log('test/bleu', results['score'], on_epoch=True, on_step=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer