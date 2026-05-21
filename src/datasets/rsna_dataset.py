
import os
import torch
import numpy as np
import pydicom

from torch.utils.data import Dataset


class RSNADicomDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        patient_id = row["patientId"]
        label = row["Target"]

        dicom_path = os.path.join(
            self.image_dir,
            patient_id + ".dcm"
        )

        dcm = pydicom.dcmread(dicom_path)

        image = dcm.pixel_array.astype(np.float32)

        image_min = image.min()
        image_max = image.max()

        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)

        image = np.stack([image] * 3, axis=-1)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor(label, dtype=torch.float32)

        return image, label
