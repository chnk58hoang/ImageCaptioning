from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as T
import torch
import os
import sentencepiece


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """

    def __init__(self, img_dir, dataframe, vocab_model):
        self.root_dir = img_dir
        self.df = dataframe
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Get image and caption colum from the dataframe
        self.vocab_model = vocab_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df['caption'].iloc[idx]
        img_name = self.df['image'].iloc[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        # apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)

        encode_captions = self.vocab_model.encode_as_ids(caption)

        return img, torch.tensor(encode_captions)


def collate_fn(batch):
    (imgs, caption_vecs) = zip(*batch)
    imgs = torch.stack([img for img in imgs], dim=0)
    caption_vecs = pad_sequence([cap for cap in caption_vecs], batch_first=True, padding_value=0)
    return imgs, caption_vecs


if __name__ == '__main__':
    pass
