from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            if filename == ".DS_Store":
                pass
            else:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return file_paths


class Logos(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.transform = transform
        self.path_to_logo_img_files = os.path.join(dataset_root, "clients_logos")
        files = get_filepaths(self.path_to_logo_img_files)
        self.logo_dataset = []
        self.labels = []
        index = 0
        self.label_to_index = {}
        for f in files:
            i1 = f.split("/")[-1]
            i1 = i1.split(".")[0]
            brand = i1.split("_")[0]
            if brand not in self.label_to_index.keys():
                self.label_to_index[brand] = index
                index += 1
            self.logo_dataset.append((f, self.label_to_index[brand]))
            self.labels.append(self.label_to_index[brand])

        self.labels = np.asarray(self.labels)

    def __len__(self):
        return len(self.logo_dataset)

    def __getitem__(self, idx):
        img_path, label = self.logo_dataset[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        output_dict = {"data": img, "label": label}
        return output_dict
