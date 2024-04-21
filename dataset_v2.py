import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

class ASL_MNIST(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        # Remap labels to be continuous
        self.label_map = {k: v for v, k in enumerate(sorted(self.data_frame['label'].unique()))}
        self.data_frame['label'] = self.data_frame['label'].map(self.label_map)
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
            ])
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_as_np = np.array(self.data_frame.iloc[idx, 1:]).astype('uint8').reshape(28, 28)
        label = int(self.data_frame.iloc[idx, 0])

        img = Image.fromarray(img_as_np, 'L')

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == '__main__':
    dataset = ASL_MNIST(csv_file=os.path.join('ASL_MNIST', 'train', 'sign_mnist.csv'))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        images, labels = batch
        print(f"Img: {images.shape}\n{images}\n\nLAb: {labels.shape}\n{labels}")
        img = images[0][0].numpy() 
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {labels[0].numpy()}')
        plt.show()
        break

