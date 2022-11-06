import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from parex import ParExModel
from transformers import AutoTokenizer
import torchmetrics

class LitParEx(pl.LightningModule):
    def __init__(
        self, extract_model_ck: str, reconstruct_model_ck: str,
        load_pretrained_mapping: bool, mapping_ck: str, tokenizer_ck: str):
        super(LitParEx, self).__init__()
        self.parex = ParExModel(
            extract_model_ck,
            reconstruct_model_ck,
            load_pretrained_mapping,
            mapping_ck
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=1)
        self.train_bleu = torchmetrics.BLEUScore()
        self.valid_bleu = torchmetrics.BLEUScore()
        self.test_bleu = torchmetrics.BLEUScore()
        self.tokenizer = ParExTokenizer(tokenizer_ck)
        self.save_hyperparameters()

    def export_model(self, path_gen, path_map):
        self.parex.save_pretrained(path_gen)
        self.parex.save_mapping(path_map)

    def training_step(self, batch, batch_idx):
        labels = batch.pop('decoder_labels')
        logits = self.parex(batch)

        preds = [pred.strip() for pred in self.tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)]
        targets = [[sent.strip()] for sent in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]
        self.train_bleu.update(preds, targets)

        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1).long()
        loss = self.loss(logits, labels)

        self.log("train/loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train/bleu_epoch', self.train_bleu.compute(), on_epoch=True, sync_dist=True)
        self.train_bleu.reset()

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('decoder_labels')
        logits = self.parex(batch)

        preds = self.tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
        targets = [[sent] for sent in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]
        self.valid_bleu.update(preds, targets)

        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1).long()
        loss = self.loss(logits, labels)

        self.log("valid/loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('valid/bleu_epoch', self.valid_bleu.compute(), on_epoch=True, sync_dist=True)
        self.valid_bleu.reset()

    def test_step(self, batch, batch_idx):
        labels = batch.pop('decoder_labels')
        logits = self.parex(batch)

        preds = self.tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
        targets = [[sent] for sent in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]
        self.test_bleu.update(preds, targets)

    def test_epoch_end(self, outputs):
        self.log('test/bleu_epoch', self.test_bleu.compute(), on_epoch=True, sync_dist=True)
        self.test_bleu.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer