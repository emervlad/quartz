import hydra
import omegaconf
import torch
import pytorch_lightning as pl

from src.model import QuartzNetCTC, logger
from src.data import ASRDataset, collate_fn

@hydra.main(config_path="conf", config_name="quarznet_5x5_ru")
def main(conf: omegaconf.DictConfig) -> None:

    model = QuartzNetCTC(conf)

    if conf.model.init_weights:
        ckpt = torch.load(conf.model.init_weights, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        logger.info("successful load initial weights")

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(save_dir="logs"), **conf.trainer
    )

    trainer.fit(model)

    # farfield

    # trainer.validate(model, dataloaders=torch.utils.data.DataLoader(
    #         ASRDataset(conf.val_dataloader.dataset),
    #         batch_size=conf.val_dataloader.batch_size,
    #         num_workers=conf.val_dataloader.num_workers,
    #         prefetch_factor=conf.train_dataloader.prefetch_factor,
    #         collate_fn=collate_fn,
    #     ))

    # crowd

    # trainer.validate(model, dataloaders=torch.utils.data.DataLoader(
    #         ASRDataset(conf.val_dataloader2.dataset),
    #         batch_size=conf.val_dataloader2.batch_size,
    #         num_workers=conf.val_dataloader2.num_workers,
    #         prefetch_factor=conf.train_dataloader.prefetch_factor,
    #         collate_fn=collate_fn,
    #     ))


if __name__ == "__main__":
    main()
