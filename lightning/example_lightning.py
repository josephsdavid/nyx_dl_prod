import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import util
from argparse import ArgumentParser


class ExampleModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.hparams["tpu_cores"] = 0
        self.loss = self.get_loss_fn()
        # you can get fancier here of course, we will likely have a separate
        # class for the model
        self.model = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.hparams.n_classes),
        )

    def forward(self, inputs):
        # defines what happens at inference times
        logits = self.model(inputs)
        probs = (
            torch.sigmoid(logits)
            if self.hparams.n_classes == 1
            else nn.Softmax(-1)(logits)
        )
        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        # see the arguments to self.log
        # https://pytorch-lightning.readthedocs.io/en/stable/new-project.html#logging
        return loss

    def training_epoch_end(self, training_step_outputs):
        example = []
        for loss in training_step_outputs:
            example.append(loss.detach().cpu().item())
        training_epoch_loss = sum(example) / len(example)
        self.log(
            "train_epoch_loss",
            training_epoch_loss,
            logger=True,
            on_step=True,
            prog_bar=True,
        )
        return training_epoch_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log("validation_loss", loss, logger=True)
        # do more things to evaluate intuitive metrics

    def validation_epoch_end(self, valid_step_outputs):
        for loss in valid_step_outputs:
            example.append(loss.detach().cpu().item())
        val_loss = sum(example) / len(example)
        self.log("val_epoch_loss", val_loss, logger=True)
        # perhaps log some sort of visualization of a transformation on the data
        # or something, you can do anything you want
        return val_loss

    def get_loss_fn(self):
        # you can either get the loss function from args or by hand
        loss = (
            nn.BCEWithLogitsLoss()
            if self.hparams.n_classes == 1
            else nn.CrossEntropyLoss()
        )
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizers == "Adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        else:
            optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
            )

        # test this
        return util.set_schedule(self, optim)

    def __dataloader(self, split):
        # write this
        pass

    def val_dataloader(self):
        return self.__dataloader("valid")

    def train_dataloader(self):
        return self.__dataloader("train")

    def test_dataloader(self):
        return self.__dataloader("test")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_classes", type=int, default=1)
