
import timm


def build_model(num_classes=1, drop_rate=0.3):

    model = timm.create_model(
        "densenet121",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=drop_rate
    )

    return model
