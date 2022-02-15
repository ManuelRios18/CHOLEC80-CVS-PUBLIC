import os
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


class Cholec80CSVDataset(Dataset):

    def __init__(self, set_name, root_dir, transform=None, predefined_set=None):
        assert set_name in ["train", "val", "test"]
        if predefined_set is None:
            self.labels_df = pd.read_csv(f"data/{set_name}.csv")
        else:
            self.labels_df = predefined_set
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        row = self.labels_df.iloc[index]
        video_name = row["video_name"]
        image_name = f"{row['image']}.jpg"
        image = io.imread(os.path.join(self.root_dir, video_name, image_name))[:, :, :3]
        two_structures_score = int(min(1,row["two_structures_score"]))
        cystic_plate_score = int(min(1, row["cystic_plate_score"]))
        hc_triangle_score = int(min(1, row["hc_triangle_score"]))
        target = np.array([two_structures_score, cystic_plate_score, hc_triangle_score])
        if self.transform:
            image = self.transform(image)
        return image, target, video_name + "/" + image_name
