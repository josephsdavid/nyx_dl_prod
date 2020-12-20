import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import util
from argparse import ArgumentParser
#from dataset import get_dataloader
from models import AutoEncoder


class AutoencoderModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.val_dict = {}
        self.train_losses=[]
        self.hparams = hparams
        self.hparams["tpu_cores"] = 0
        self.loss = self.get_loss_fn()
        # you can get fancier here of course, we will likely have a separate
        # class for the model
        self.model = AutoEncoder(
            self.hparams["latent_dim"]
        )

    def forward(self, inputs):
        output = self.model(inputs)
        return dict(latent=output[0], predicted=output[1])

    def training_step(self, batch, batch_idx):
        x = batch
        sign = torch.sign(x)
        _, preds = self.model(x)
        if self.hparams['sigmoid']:
            preds = torch.sigmoid(preds)
        preds = preds * sign
        loss = self.loss(preds, x)
        self.train_losses.append(loss.detach().cpu().item())
        self.log(
                "train_loss",
                loss,
                on_epoch=True,
                on_step=True,
                logger=True,
                prog_bar=True,
                )

        return loss

    def training_epoch_end(self, training_result):
        self.log(
                "epoch_train_loss",
                sum(self.train_losses)/len(self.train_losses),
                on_epoch=True,
                logger=True,
                )
        self.train_losses = []

    def validation_step(self, batch, batch_idx):
        x = batch
        sign = torch.sign(x)
        _, preds = self.model(x)
        if self.hparams['sigmoid']:
            preds = torch.sigmoid(preds)
        loss = self.loss(preds * sign, x)
        self.log("val_loss", loss, on_epoch=True, on_step=False, )
        for n in [1, 5, 10, 20]:
            x_mask = x.clone().detach()
            for i in range(x_mask.shape[0]):
                num_revs = x_mask[i, :].bool().sum()
                if n > num_revs:
                    x_mask[i, :] = 0
                else:
                    x_mask[i, :][torch.where(x_mask[i, :] > 0)[-n:]] = 0
            _, preds = self.model(x_mask)
            if self.hparams['sigmoid']:
                preds = torch.sigmoid(preds)
            loss = self.loss(preds * sign, x)
            self.log(f"val_last_{n}_loss", loss, on_epoch=True, on_step=False, )
            self.val_dict.setdefault(f"val_last_{n}_loss", []).append(loss.detach().cpu().item())

    def validation_epoch_end(self, validation_result):
        for k, v in self.val_dict.items():
            self.log(f"epoch_{k}", sum(v) / len(v), on_epoch=True, logger=True)
        self.val_dict = {}

    def get_loss_fn(self):
        if self.hparams['reduction'] == "sum":
            loss = nn.MSELoss(reduction='sum')
        else:
            loss = nn.MSELoss()
        final_loss = loss
        return final_loss

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "Adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])
        else:
            optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams["lr"],
                momentum=self.hparams["momentum"],
            )

        # test this
        return util.set_schedule(self, optim)

    #def __dataloader(self, split):
    #    return get_dataloader(split, self.hparams)

    #def val_dataloader(self):
    #    return self.__dataloader("valid")

    #def train_dataloader(self):
    #    return self.__dataloader("train")

    #def test_dataloader(self):
    #    return self.__dataloader("test")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--latent_dim", type=int, default=256)
        parser.add_argument("--scheduler", type=str, default="none")
        parser.add_argument("--reduction", type=str, default="mean")
        parser.add_argument("--normalize", type=util.str2bool, default=False)
        parser.add_argument("--sigmoid", type=util.str2bool, default=False)
        return parser
