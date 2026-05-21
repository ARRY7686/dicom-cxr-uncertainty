
import albumentations as A

from albumentations.pytorch import ToTensorV2


def get_train_transforms():

    return A.Compose([

        A.Resize(224, 224),

        A.HorizontalFlip(p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),

        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),

        ToTensorV2()
    ])


def get_val_transforms():

    return A.Compose([

        A.Resize(224, 224),

        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),

        ToTensorV2()
    ])
