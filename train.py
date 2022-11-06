from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from parex import ParExDataLoader, LitParEx

if __name__ == "__main__":
    # wandb_logger = WandbLogger(project="proj_parex")
    # wandb_logger = WandbLogger(project="proj_dummy")

    # model
    extract_ckpt = 'sentence-transformers/all-MiniLM-L6-v2'
    gen_ckpt = 'facebook/bart-base'
    load_pretrained_mapping = False
    mapping_ck = ""
    tokenizer_ck = 'parex_tokenizer'

    lit_parex = LitParEx(
        extract_ckpt,
        gen_ckpt,
        load_pretrained_mapping,
        mapping_ck,
        tokenizer_ck=gen_ckpt
    )

    # dataloader
    parex_dataloader = ParExDataLoader(gen_ckpt, extract_ckpt, 80, 40)
    [train_dataloader, valid_dataloader, test_dataloader] = parex_dataloader.get_dataloader(batch_size=64, types=["train", "valid", "test"])

    # train model
    # trainer = pl.Trainer(max_epochs=2, logger=wandb_logger, devices=[0], accelerator="gpu")#, strategy="ddp")
    trainer = pl.Trainer(max_epochs=2, accelerator="cpu")
    trainer.fit(model=lit_parex, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # # save model & tokenizer
    # lit_parex.export_model('parex_model/v3/gen', 'parex_model/v3/map')
