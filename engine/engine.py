import pytorch_lightning as pl
from torch.utils.data import RandomSampler, DataLoader
from dataset.dataset import collate_fn


def calculate_blue(pred, cap):
    pass

def visualizeimage():
    pass


class MyCallbacks(pl.Callback):
    def __init__(self, test_dataset, vocab_model):
        super(MyCallbacks, self).__init__()
        self.testdataset = test_dataset
        self.vocab_model = vocab_model

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        subset = RandomSampler(data_source=self.testdataset, num_samples=3)
        dataloader = DataLoader(dataset=self.testdataset, batch_size=1, sampler=subset, collate_fn=collate_fn)

        all_preds = []
        all_caps = []

        for batch_idx, data in enumerate(dataloader):
            image = data[0]
            caption = data[1].tolist()

            pred_caption = pl_module.predict(image)
            pred_caption = self.vocab_model.decode_ids(pred_caption)

            caption = self.vocab_model.decode_ids(caption)
            print("Prediction:{}".format(pred_caption))
            print("Label: {}".format(caption))
            print('\n')
