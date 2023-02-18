from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from ParEx import ParExDataLoader, LitParEx

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_parex")

    # model
    extract_ckpt = 'sentence-transformers/all-MiniLM-L6-v2'
    gen_ckpt = 'new_bart'
    tokenizer_ck = 'facebook/bart-base'

    lit_parex = LitParEx(extract_ckpt, gen_ckpt, tokenizer_ck, lr=5e-5, num_beams=4, max_length=128)

    # dataloader
    parex_dataloader = ParExDataLoader(tokenizer_ck, extract_ckpt, 128, 64)
    [train_dataloader, valid_dataloader, test_dataloader] = parex_dataloader.get_dataloader(batch_size=32, types=["train", "valid", "test"])

    # train model
    trainer = pl.Trainer(max_epochs=20, devices=[0], accelerator="gpu", val_check_interval=2000, logger=wandb_logger)#, strategy="ddp")
    trainer.fit(model=lit_parex, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(dataloaders=test_dataloader)

    # save model & tokenizer
    lit_pare.export_model('parex_model/v1')
