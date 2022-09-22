import argparse
import pandas as pd
import pytorch_lightning as pl
import sentencepiece as spm
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from dataset.dataset import FlickrDataset, collate_fn
from modules.model import ImageCaption
from torch.utils.data import DataLoader
from engine.engine import MyCallbacks


class TrainModule(pl.LightningModule):
    def __init__(self, model, device, train_loader, valid_loader, vocab_size):
        super(TrainModule, self).__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.device_ = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.vocab_size = vocab_size

    def forward(self, image, caption):
        image = image.to(self.device_)
        caption = caption.to(self.device_)
        return self.model(image, caption)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        # return {'optimizer':self.optimizer,'sheduler':self.scheduler,'metric':"val_loss"}

        return [self.optimizer]

    def training_step(self, batch, batch_idx):
        image, caption = batch
        out, attention = self.forward(image, caption)
        label = caption[:, 1:].reshape(-1)
        out = out.view(-1, self.vocab_size)
        loss = self.loss(out, label)
        tensorboard_logs = {'train_loss': loss}
        return {"loss": loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        image, caption = batch
        out, attention = self.forward(image, caption)
        label = caption[:, 1:].reshape(-1)
        out = out.view(-1, self.vocab_size)
        loss = self.loss(out, label)
        tensorboard_logs = {'val_loss': loss}
        return {"loss": loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def predict(self, image):
        image = image.to(self.device_)
        return self.model.predict(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='archive/images')
    parser.add_argument('--vocab', type=str, default='m.model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ft', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--csv', type=str, default='data.csv')
    args = parser.parse_args()

    """Read dataframe from csv"""
    df = pd.read_csv(args.csv)
    df_len = len(df)

    """Split dataframe to train, val and test set"""
    df = df.sample(frac=1)
    train_df_len = int(df_len * 0.70)
    val_df_len = int(df_len * 0.85)

    train_df = df[:train_df_len]
    val_df = df[train_df_len:val_df_len]
    test_df = df[val_df_len:]

    """Vocab_model"""
    sp = spm.SentencePieceProcessor()
    sp.load(args.vocab)

    """Create dataset and dataloader"""

    train_dataset = FlickrDataset(dataframe=train_df, img_dir=args.img, vocab_model=sp)
    val_dataset = FlickrDataset(dataframe=val_df, img_dir=args.img, vocab_model=sp)
    test_dataset = FlickrDataset(dataframe=test_df, img_dir=args.img, vocab_model=sp)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    # test_dataloader = DataLoader(test_dataset, batch_size=3, collate_fn=collate_fn)

    """Device"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ImageCaption(embed_size=300, vocab_size=200, attention_dim=256, encoder_dim=2048,
                         decoder_dim=512, device=device).to(device)

    """Trainer and Callback"""
    train_module = TrainModule(model=model, device=device, train_loader=train_dataloader, valid_loader=val_dataloader,
                               vocab_size=200)
    callbacks = MyCallbacks(test_dataset=test_dataset, vocab_model=sp)
    trainer = Trainer(max_epochs=args.epochs, callbacks=[callbacks, ],) #accelerator='gpu', gpus=1)
    trainer.fit(train_module)
